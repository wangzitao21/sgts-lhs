import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

import shap
from sko.GA import GA
import seaborn as sns

from smt.sampling_methods import LHS

from models.sampler import SGTS_LHS
config = {'font.family': 'Arial',
          'font.size': 13,
          'xtick.direction': 'in',
          'ytick.direction': 'in',
          'mathtext.fontset': 'stix',
          'savefig.dpi': 300
         }
plt.rcParams.update(config)

# =================================================
# Define the objective function
# =================================================
DEFAULT_WEIGHTS = np.array([10.0, 8.0, 5.0, 0.5, 0.3, 0.2, 0.1]) # Default weights

def weighted_ackley_function(x_vector, weights=None, a=20, b=0.2, c=2 * np.pi):
    """Calculate the value of the weighted Ackley function."""
    n_local = len(x_vector)
    if weights is None:
        if n_local == len(DEFAULT_WEIGHTS): current_weights = DEFAULT_WEIGHTS
        else: current_weights = np.ones(n_local)
    elif len(weights) != n_local: raise ValueError("The length of the input vector x_vector and the weight vector weights must be the same.")
    else: current_weights = np.array(weights)

    weighted_x_sq = (x_vector * current_weights)**2
    weighted_x_cos = x_vector * current_weights
    sum_sq_term = -b * np.sqrt(np.sum(weighted_x_sq) / n_local)
    sum_cos_term = np.sum(np.cos(c * weighted_x_cos)) / n_local
    result = -a * np.exp(sum_sq_term) - np.exp(sum_cos_term) + a + np.exp(1)
    return result

# =================================================
# SGTS-LHS Parameter Settings
# =================================================
n_dim = 7
search_bound = 5.0
param_names = [f'p{i+1}' for i in range(n_dim)]

n_samples_s1_factor = 100  # Number of samples factor per dimension for Stage 1
n_samples_s2_factor = 10   # Number of samples factor per dimension for Stage 2 (if <=0, SGTS-LHS performs single-stage sampling)

k_best_for_roi_bounds_factor = 0.01  # Percentage of best samples from Stage 1 used to determine ROI boundaries
roi_padding_factor_important = 0.01  # Boundary padding factor for important features in the ROI
roi_padding_factor_non_important = 0.02  # Boundary padding factor for non-important features in the ROI

# Preliminary surrogate model and SHAP analysis parameters
preliminary_surrogate_model_type_for_shap = 'rf' # Type of the preliminary surrogate model

n_top_shap_features_for_roi = 3  # Number of top SHAP features for ROI optimization

# Evaluation Parameters
n_test_eval_factor = 2000  # Factor for the number of test points (n_dim * factor)

# Experiment Control
n_runs = 50  # Number of times to run the entire experiment to ensure robustness
master_seed = 2025 # Master seed for experiment reproducibility

# =================================================
# Final Surrogate Model Training Function
# =================================================
def train_final_surrogate_model(x_train_data, y_train_data, model_type='rf', model_params=None, random_seed=None):
    """Train the final surrogate model."""
    if model_params is None: model_params = {}
    if model_type == 'rf':
        rf_params = {
            'n_estimators': model_params.get('n_estimators', 200),
            'max_depth': model_params.get('max_depth', None),
            'min_samples_split': model_params.get('min_samples_split', 2),
            'min_samples_leaf': model_params.get('min_samples_leaf', 1),
            'n_jobs': model_params.get('n_jobs', -1),
            'random_state': random_seed
        }
        model = RandomForestRegressor(**rf_params)
        model.fit(x_train_data, y_train_data.ravel())
        return model
    else:
        raise ValueError(f"Unsupported final model type: {model_type}")

# =================================================
# Model Evaluation Helper Function
# =================================================
def evaluate_surrogate_model(model, x_test_data, y_test_true_data):
    """Evaluate the performance of the surrogate model."""
    y_pred_flat = model.predict(x_test_data).ravel()
    y_test_true_flat = y_test_true_data.ravel()
    rmse = np.sqrt(mean_squared_error(y_test_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_test_true_flat, y_pred_flat)
    r2 = r2_score(y_test_true_flat, y_pred_flat)
    return {"rmse": rmse, "mae": mae, "r2": r2, "y_pred": y_pred_flat}

print(f"--- Experiment Configuration ---")
print(f"Problem dimension: {n_dim}, Search bounds: [-{search_bound}, {search_bound}]")

# Determine if SGTS-LHS is single-stage or two-stage based on n_samples_s2_factor, and calculate the total budget
sgts_is_two_stage = n_samples_s2_factor > 0
if sgts_is_two_stage:
    sgts_total_samples_budget = (n_samples_s1_factor * n_dim) + (n_samples_s2_factor * n_dim)
    print(f"SGTS-LHS performs two-stage sampling, total sample budget: {sgts_total_samples_budget}")
    print(f"  (S1: {n_samples_s1_factor * n_dim}, S2: {n_samples_s2_factor * n_dim})")
else:
    sgts_total_samples_budget = n_samples_s1_factor * n_dim # The S1 factor now represents the total factor
    print(f"SGTS-LHS performs single-stage sampling, total sample budget: {sgts_total_samples_budget}")
print(f"Number of experiment runs: {n_runs}")

lb_global_setup = np.array([-search_bound] * n_dim)
ub_global_setup = np.array([search_bound] * n_dim)
xlimits_global_setup = np.vstack((lb_global_setup, ub_global_setup)).T

print("\n--- Generating a consistent evaluation test set ---")
n_test_points_eval = n_test_eval_factor * n_dim
test_set_seed = master_seed + 701 
x_test_shared_eval = LHS(xlimits=xlimits_global_setup, criterion='c', random_state=test_set_seed)(n_test_points_eval)
y_test_true_shared_eval = np.array([weighted_ackley_function(x) for x in x_test_shared_eval]).reshape(-1, 1)
print(f"Shared test set generated ({n_test_points_eval} points), seed {test_set_seed}.")

# Prepare containers
all_runs_results_sgts = []
all_runs_results_single_baseline = []
all_y_pred_sgts_runs = []
all_y_pred_single_baseline_runs = []
last_run_shap_plot_details_for_main = None

inversion_results_sgts = []
inversion_results_baseline = []

true_optimal_params = np.zeros(n_dim)

param_metrics_sgts = []
param_metrics_baseline = []

# Final surrogate model parameters
final_surrogate_model_params_to_use = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'n_jobs': -1
}

# Seed offsets for each stage
offset_s1, offset_prelim, offset_s2, offset_final_sgts, offset_lhs, offset_final_base = 101, 201, 301, 401, 501, 601

for i_run in range(n_runs):
    t0 = time.time()
    base = master_seed + i_run * 1000

    # SGTS-LHS sampling
    sampler = SGTS_LHS(
        n_dim=n_dim,
        param_names=param_names,
        x_limits_global=xlimits_global_setup,
        objective_func=weighted_ackley_function,
        n_samples_s1_factor=n_samples_s1_factor,
        n_samples_s2_factor=n_samples_s2_factor,
        k_best_for_roi_bounds_factor=k_best_for_roi_bounds_factor,
        roi_padding_factor_important=roi_padding_factor_important,
        roi_padding_factor_non_important=roi_padding_factor_non_important,
        preliminary_surrogate_model_type=preliminary_surrogate_model_type_for_shap,
        preliminary_surrogate_model_params={},
        n_top_shap_features=n_top_shap_features_for_roi,
        s1_lhs_seed=base + offset_s1,
        preliminary_model_train_seed=base + offset_prelim,
        s2_lhs_roi_seed=base + offset_s2
    )
    x_sgts, y_sgts, calls_sgts, shap_details = sampler.generate_samples()

    # SGTS final model training and evaluation
    model_sgts = train_final_surrogate_model(
        x_sgts, y_sgts,
        model_type='rf',
        model_params=final_surrogate_model_params_to_use,
        random_seed=base + offset_final_sgts
    )
    metrics_sgts = evaluate_surrogate_model(model_sgts, x_test_shared_eval, y_test_true_shared_eval)
    metrics_sgts['calls'] = calls_sgts
    all_runs_results_sgts.append(metrics_sgts)
    all_y_pred_sgts_runs.append(metrics_sgts['y_pred'])
    last_run_shap_plot_details_for_main = shap_details

    print(f"Run {i_run+1}/{n_runs} SGTS: RMSE={metrics_sgts['rmse']:.4f}, MAE={metrics_sgts['mae']:.4f}, R2={metrics_sgts['r2']:.4f}")

    # ## Inversion ##
    # ga_sgts = GA(
    #     func=lambda p: model_sgts.predict(np.array(p).reshape(1, -1)).item(),
    #     n_dim=n_dim, size_pop=200, max_iter=100,
    #     lb=[-search_bound] * n_dim, ub=[search_bound] * n_dim, precision=1e-6
    # )
    # best_params_sgts, _ = ga_sgts.run()
    # true_val_at_best_sgts = weighted_ackley_function(best_params_sgts)
    # inversion_results_sgts.append({'true_val_at_best': true_val_at_best_sgts})
    # print(f"  >>> SGTS Inversion Result: True value at best point found = {true_val_at_best_sgts:.6f}")
    # param_mse = mean_squared_error(true_optimal_params, best_params_sgts)
    # param_mae = mean_absolute_error(true_optimal_params, best_params_sgts)
    # param_r2 = r2_score(true_optimal_params, best_params_sgts)
    # param_metrics_sgts.append({'mse': param_mse, 'mae': param_mae, 'r2': param_r2})
    # print(f"  >>> SGTS Inversion PARAMETER Quality: MSE={param_mse:.4f}, MAE={param_mae:.4f}, R2={param_r2:.4f}")

    # Single-stage LHS sampling
    x_base = LHS(xlimits=xlimits_global_setup, criterion='m', random_state=base + offset_lhs)(sgts_total_samples_budget)
    y_base = np.array([weighted_ackley_function(x) for x in x_base]).reshape(-1, 1)
    calls_base = x_base.shape[0]

    # Baseline model training and evaluation
    model_base = train_final_surrogate_model(
        x_base, y_base,
        model_type='rf',
        model_params=final_surrogate_model_params_to_use,
        random_seed=base + offset_final_base
    )
    metrics_base = evaluate_surrogate_model(model_base, x_test_shared_eval, y_test_true_shared_eval)
    metrics_base['calls'] = calls_base
    all_runs_results_single_baseline.append(metrics_base)
    all_y_pred_single_baseline_runs.append(metrics_base['y_pred'])

    print(f"Run {i_run+1}/{n_runs} Baseline: RMSE={metrics_base['rmse']:.4f}, MAE={metrics_base['mae']:.4f}, R2={metrics_base['r2']:.4f}")

    # ## Inversion ##
    # ga_base = GA(
    #     func=lambda p: model_base.predict(np.array(p).reshape(1, -1)).item(),
    #     n_dim=n_dim, size_pop=200, max_iter=100,
    #     lb=[-search_bound] * n_dim, ub=[search_bound] * n_dim, precision=1e-6
    # )
    # best_params_base, _ = ga_base.run()
    # true_val_at_best_base = weighted_ackley_function(best_params_base)
    # inversion_results_baseline.append({'true_val_at_best': true_val_at_best_base})
    # print(f"  >>> Baseline Inversion Result: True value at best point found = {true_val_at_best_base:.6f}")
    # param_mse = mean_squared_error(true_optimal_params, best_params_base)
    # param_mae = mean_absolute_error(true_optimal_params, best_params_base)
    # param_r2 = r2_score(true_optimal_params, best_params_base)
    # param_metrics_baseline.append({'mse': param_mse, 'mae': param_mae, 'r2': param_r2})
    # print(f"  >>> Baseline Inversion PARAMETER Quality: MSE={param_mse:.4f}, MAE={param_mae:.4f}, R2={param_r2:.4f}")

    print(f"  Time: {time.time() - t0:.2f} sec\n")

# Summarize and print results
calls_s = np.array([r['calls'] for r in all_runs_results_sgts])
rmse_s = np.array([r['rmse']  for r in all_runs_results_sgts])
mae_s  = np.array([r['mae']   for r in all_runs_results_sgts])
r2_s   = np.array([r['r2']    for r in all_runs_results_sgts])

calls_b = np.array([r['calls'] for r in all_runs_results_single_baseline])
rmse_b = np.array([r['rmse']  for r in all_runs_results_single_baseline])
mae_b  = np.array([r['mae']   for r in all_runs_results_single_baseline])
r2_b   = np.array([r['r2']    for r in all_runs_results_single_baseline])

print(f"\nSGTS-LHS Average calls: {calls_s.mean():.0f} ±{calls_s.std():.2f}")
print(f"SGTS-LHS RMSE: {rmse_s.mean():.4f} ±{rmse_s.std():.4f} (min {rmse_s.min():.4f}, max {rmse_s.max():.4f})")
print(f"SGTS-LHS MAE:  {mae_s.mean():.4f} ±{mae_s.std():.4f} (min {mae_s.min():.4f}, max {mae_s.max():.4f})")
print(f"SGTS-LHS R2:   {r2_s.mean():.4f} ±{r2_s.std():.4f} (min {r2_s.min():.4f}, max {r2_s.max():.4f})\n")

print(f"Baseline Average calls: {calls_b.mean():.0f} ±{calls_b.std():.2f}")
print(f"Baseline RMSE: {rmse_b.mean():.4f} ±{rmse_b.std():.4f} (min {rmse_b.min():.4f}, max {rmse_b.max():.4f})")
print(f"Baseline MAE:  {mae_b.mean():.4f} ±{mae_b.std():.4f} (min {mae_b.min():.4f}, max {mae_b.max():.4f})")
print(f"Baseline R2:   {r2_b.mean():.4f} ±{r2_b.std():.4f} (min {r2_b.min():.4f}, max {r2_b.max():.4f})")

# =================================================

# sgts_inversion_vals = [r['true_val_at_best'] for r in inversion_results_sgts]
# base_inversion_vals = [r['true_val_at_best'] for r in inversion_results_baseline]

# print("\n--- SHAP-Guided Two-Stage Strategy (Inversion Performance) ---")
# print(f"  True function value at the optimum found by GA:")
# print(f"    Mean: {np.mean(sgts_inversion_vals):.4f}, Standard Deviation: {np.std(sgts_inversion_vals):.4f}")
# print(f"    Minimum (best performance): {np.min(sgts_inversion_vals):.4f}")

# print("\n--- Single-Stage Baseline Strategy (Inversion Performance) ---")
# print(f"  True function value at the optimum found by GA:")
# print(f"    Mean: {np.mean(base_inversion_vals):.4f}, Standard Deviation: {np.std(base_inversion_vals):.4f}")
# print(f"    Minimum (best performance): {np.min(base_inversion_vals):.4f}")

# ## --- Add new summary section: Parameter Inversion Accuracy --- ##
# print("\n\n--- Parameter Inversion Accuracy Summary (Difference between found parameters and true optimal parameters [0,0,...]) ---")

# # Extract parameter metrics for SGTS
# sgts_param_mses = [r['mse'] for r in param_metrics_sgts]
# sgts_param_maes = [r['mae'] for r in param_metrics_sgts]
# sgts_param_r2s  = [r['r2'] for r in param_metrics_sgts] # <<< Extract R2 results

# # Extract parameter metrics for Baseline
# base_param_mses = [r['mse'] for r in param_metrics_baseline]
# base_param_maes = [r['mae'] for r in param_metrics_baseline]
# base_param_r2s  = [r['r2'] for r in param_metrics_baseline] # <<< Extract R2 results

# print("\n--- SHAP-Guided Two-Stage Strategy (Parameter Inversion Accuracy) ---")
# print(f"  Parameter MSE:  Avg={np.mean(sgts_param_mses):.4f}, Std={np.std(sgts_param_mses):.4f}")
# print(f"  Parameter MAE: Avg={np.mean(sgts_param_maes):.4f}, Std={np.std(sgts_param_maes):.4f}")
# print(f"  Parameter R2:  Avg={np.mean(sgts_param_r2s):.4f}, Std={np.std(sgts_param_r2s):.4f}") # <<< Add R2 statistics

# print("\n--- Single-Stage Baseline Strategy (Parameter Inversion Accuracy) ---")
# print(f"  Parameter MSE:  Avg={np.mean(base_param_mses):.4f}, Std={np.std(base_param_mses):.4f}")
# print(f"  Parameter MAE:  Avg={np.mean(base_param_maes):.4f}, Std={np.std(base_param_maes):.4f}")
# print(f"  Parameter R2:   Avg={np.mean(base_param_r2s):.4f}, Std={np.std(base_param_r2s):.4f}") # <<< Add R2 statistics


# =================================================
# Plot 1
# =================================================
n = len(DEFAULT_WEIGHTS)
x_range = np.linspace(-5, 5, 400)
base_x = np.zeros(n)

plt.figure(figsize=(6, 4))
ax = plt.gca() # Get the current axes

for i, w in enumerate(DEFAULT_WEIGHTS):
    # Calculate function values for different values of the variable in the i-th dimension
    y_values = []
    for val in x_range:
        x_vec = base_x.copy()
        x_vec[i] = val
        y_values.append(weighted_ackley_function(x_vec))
    # Plot in the same axes
    ax.plot(x_range, y_values, color=f'C{i}', label=f'x$_{i}$, w={w}')

ax.set_xlabel('x')
ax.set_ylabel('f(x, y)')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()

# =================================================
# Plot 2
# =================================================
x = np.linspace(-1, 1, 300)
y = np.linspace(-1, 1, 300)
X, Y = np.meshgrid(x, y)

weights1 = [1.0, 1.0]
Z1 = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z1[i, j] = weighted_ackley_function(np.array([X[i, j], Y[i, j]]), weights=weights1)

weights2 = [10.0, 1.0]
Z2 = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z2[i, j] = weighted_ackley_function(np.array([X[i, j], Y[i, j]]), weights=weights2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

contour1 = ax1.contourf(X, Y, Z1, levels=20, cmap='viridis')
ax1.contour(X, Y, Z1, levels=20, colors='white', linewidths=0.5)
ax1.set_xlabel('x0')
ax1.set_ylabel('x1')
ax1.set_aspect('equal', adjustable='box')

contour2 = ax2.contourf(X, Y, Z2, levels=20, cmap='viridis')
ax2.contour(X, Y, Z2, levels=20, colors='white', linewidths=0.5)
ax2.set_xlabel('x0')
ax2.set_ylabel('x1')
ax2.set_aspect('equal', adjustable='box')

fig.colorbar(contour2, ax=[ax1, ax2], orientation='vertical', label='Function Value f(x)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# =================================================
# Plot 3
# =================================================
metrics_to_plot = [
    {'data_b': rmse_b, 'data_s': rmse_s, 'name': 'MSE'},
    {'data_b': mae_b,  'data_s': mae_s,  'name': 'MAE'},
    {'data_b': r2_b,   'data_s': r2_s,   'name': 'R²'}
]

fig, axes = plt.subplots(1, 3, figsize=(7, 4))
custom_palette = ['#EB7E60', '#1f77b4', '#2ca02c']
palette = custom_palette[:2]

for ax, metric_info in zip(axes, metrics_to_plot):
    
    data_b = metric_info['data_b']
    data_s = metric_info['data_s']
    metric_name = metric_info['name']

    df_for_plot = pd.DataFrame({
        metric_name: np.concatenate([data_b, data_s]),
        'Method': ['LHS'] * len(data_b) + ['SGTS-LHS'] * len(data_s)
    })

    sns.boxplot(
        data=df_for_plot,
        x='Method',
        y=metric_name,
        ax=ax,
        showfliers=False,
        boxprops={'edgecolor': 'black', 'alpha': 0.8, 'facecolor': 'None'},
        whiskerprops={'color': 'black', 'alpha': 0.8},
        capprops={'color': 'black', 'alpha': 0.8},
        medianprops={'color': 'red', 'linewidth': 1.5},
        width=0.4,
    )

    sns.stripplot(
        data=df_for_plot,
        x='Method',
        y=metric_name,
        ax=ax,
        alpha=0.6,
        size=5,
        jitter=0.1,
        palette=palette
    )
    
    ax.grid(visible=True, which='major', axis='both', alpha=0.5, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlabel('Method', fontsize=13)
    ax.set_ylabel(metric_name, fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()

# =================================================
# Plot 4
# =================================================
fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 4),
    # gridspec_kw={'width_ratios': [1, 2]}
)

plt.sca(axes[0])
shap.summary_plot(
    shap_details['shap_values'],
    shap_details['x_samples'],
    feature_names=shap_details['feature_names'],
    plot_type="bar",
    show=False
)

# Second subplot
plt.sca(axes[1])
shap.summary_plot(
    shap_details["shap_values"],
    shap_details["x_samples"],
    feature_names=shap_details["feature_names"],
    cmap="jet",
    show=False
)

plt.subplots_adjust(wspace=0.4)
plt.show()

# =================================================
# Plot 5
# =================================================
fig, ax = plt.subplots(figsize=(3.1, 2.4))

shap.dependence_plot(
    "p1", 
    shap_details['shap_values'], 
    shap_details['x_samples'], 
    feature_names=shap_details['feature_names'],
    interaction_index=None ,
    ax=ax,
    show=False,
    cmap="jet",
    color="#8FB4DC",
    # edgecolors='black',
)
if ax.collections:
    ax.collections[0].set_edgecolors('black')
    ax.collections[0].set_linewidths(0.5)
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color('black')
    ax.spines[spine].set_linewidth(1)
ax.set_ylabel(None)
ax.grid(visible=True, which='major', axis='both', alpha=0.5, linestyle='--')
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()

# =================================================
# Plot 6
# =================================================
avg_sgts = np.mean(all_y_pred_sgts_runs, axis=0)
std_sgts = np.std(all_y_pred_sgts_runs, axis=0)
avg_base = np.mean(all_y_pred_single_baseline_runs, axis=0)
std_base = np.std(all_y_pred_single_baseline_runs, axis=0)
y_true = y_test_true_shared_eval.ravel()
rmse_sgts = np.sqrt(np.mean((y_true - avg_sgts)**2))
rmse_base = np.sqrt(np.mean((y_true - avg_base)**2))

vals = np.concatenate([y_true, avg_sgts, avg_sgts-std_sgts, avg_sgts+std_sgts, avg_base, avg_base-std_base, avg_base+std_base])
mn, mx = vals.min(), vals.max()
margin = (mx - mn) * 0.05 or 0.1
lo, hi = mn - margin, mx + margin
idx = np.argsort(y_true)
y_sorted, sgts_a, sgts_s = y_true[idx], avg_sgts[idx], std_sgts[idx]
base_a, base_s = avg_base[idx], std_base[idx]

fig, ax = plt.subplots(figsize=(12, 4))

perfect_line, = ax.plot([lo, hi], [lo, hi], 'k--', lw=2, label='Perfect Prediction')
sgts_points = ax.scatter(y_true, avg_sgts, c='#EB7E60', marker='o', s=35, alpha=0.7, zorder=3, label=f'SGTS-LHS (RMSE={rmse_sgts:.3f})')
ax.fill_between(y_sorted, sgts_a - sgts_s, sgts_a + sgts_s, color='#EB7E60', alpha=0.2, label=f'Uncertainty Band (SGTS-LHS)')
base_points = ax.scatter(y_true, avg_base, c='#1f77b4', marker='x', s=35, alpha=0.7, zorder=2, label=f'LHS (RMSE={rmse_base:.3f})')
ax.fill_between(y_sorted, base_a - base_s, base_a + base_s, color='#1f77b4', alpha=0.2, label=f'Uncertainty Band (LHS)')

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel('True Values', fontsize=12)
ax.set_ylabel('Predicted Values', fontsize=12)
ax.grid(linestyle=':', alpha=0.7)

ax.legend(loc='best', fontsize='medium')
ax.tick_params(axis='both', which='major', labelsize=12)

# plt.tight_layout()
plt.show()

## =================================================
## Plot 7
## =================================================
# baseline_metrics = np.load("./output/case2/baseline_metrics.npy")
# sgts_metrics = np.load("./output/case2/sgts_metrics.npy")

# metrics_to_plot = [
#     {'data_b': baseline_metrics[:, 0], 'data_s': sgts_metrics[:, 0], 'name': 'RMSE'},
#     {'data_b': baseline_metrics[:, 1], 'data_s': sgts_metrics[:, 1], 'name': 'MAE'},
#     {'data_b': baseline_metrics[:, 2], 'data_s': sgts_metrics[:, 2], 'name': 'R²'}
# ]

# fig, axes = plt.subplots(1, 3, figsize=(7, 4))
# custom_palette = ['#2ca02c', "purple"]
# palette = custom_palette[:2]

# for ax, metric_info in zip(axes, metrics_to_plot):
    
#     data_b = metric_info['data_b']
#     data_s = metric_info['data_s']
#     metric_name = metric_info['name']

#     df_for_plot = pd.DataFrame({
#         metric_name: np.concatenate([data_b, data_s]),
#         'Method': ['LHS'] * len(data_b) + ['SGTS-LHS'] * len(data_s)
#     })

#     sns.boxplot(
#         data=df_for_plot,
#         x='Method',
#         y=metric_name,
#         ax=ax,
#         showfliers=False,
#         boxprops={'edgecolor': 'black', 'alpha': 0.8, 'facecolor': 'None'},
#         whiskerprops={'color': 'black', 'alpha': 0.8},
#         capprops={'color': 'black', 'alpha': 0.8},
#         medianprops={'color': 'red', 'linewidth': 1.5},
#         width=0.4,
#     )

#     sns.stripplot(
#         data=df_for_plot,
#         x='Method',
#         y=metric_name,
#         ax=ax,
#         alpha=0.6,
#         size=5,
#         jitter=0.1,
#         palette=palette
#     )
    
#     ax.grid(visible=True, which='major', axis='both', alpha=0.5, linestyle='--')
#     ax.set_axisbelow(True)
#     ax.set_xlabel('Method', fontsize=12)
#     ax.set_ylabel(metric_name, fontsize=12)
#     ax.tick_params(axis='both', which='major', labelsize=12)

# plt.tight_layout()
# plt.show()