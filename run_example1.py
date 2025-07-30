import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sko.GA import GA
from smt.surrogate_models import KRG
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
# Define the multi-modal objective function f(x,y)
# =================================================
def f(x, y):
    f1_xy  = np.exp(-5 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
    f1_x05 = np.exp(-5 * ((x - 0.5) ** 2 + (0.5 - 0.5) ** 2))
    f2     = np.where(x + y >= 1.5, 10.0, 0.0)
    t1     = 5 * np.pi * (x - 0.5) * f1_xy
    t2     = 7 * np.pi * (x + y - 1) * f1_x05
    return 100 - 45 * f1_xy * (np.cos(t1) + np.cos(t2)) + f2
print("Reference value of the true optimum (f(0.5,0.5)):", f(0.5, 0.5))

lb_global = np.array([0.0, 0.0])
ub_global = np.array([1.0, 1.0])
xlimits_global = np.vstack((lb_global, ub_global)).T
param_names = ['x', 'y']
n_dim = len(param_names)

# =================================================
# SGTS-LHS Parameter Settings
# =================================================
num_simulations  = 50
n_samples_stage1 = 50 # Stage 1
n_samples_stage2 = 25 # Stage 2

factor_s1 = n_samples_stage1 // n_dim
factor_s2 = n_samples_stage2 // n_dim
k_best_for_roi_bounds       = 5
k_best_factor               = k_best_for_roi_bounds / n_samples_stage1
roi_padding_factor_imp      = 0.10
roi_padding_factor_nonimp   = 0.20
n_sgts_important_features   = 1

# =================================================
# Final Inversion Algorithm (GA) Parameters
# =================================================
ga_pop_size  = 50
ga_max_iter  = 100
n_samples_total = n_samples_stage1 + n_samples_stage2

# =================================================
# Pre-generate a fixed test set for RMSE evaluation
# =================================================
n_test_points   = 500
test_lhs_fixed  = LHS(xlimits=xlimits_global, criterion='m', random_state=999)
X_test_fixed    = test_lhs_fixed(n_test_points)
Y_test_true_fix = np.array([f(xv, yv) for xv, yv in X_test_fixed]).reshape(-1, 1)

# =================================================
# Results placeholder dict
# =================================================
def init_results_dict():
    return dict(best_params=[], pred_vals_at_best=[], true_vals_at_best=[], rmse=[], ga_histories=[])
results_sgts = init_results_dict()
results_1stage = init_results_dict()

last_sim_failed = False

# =================================================
# Main Simulation Loop
# =================================================
for i_sim in range(num_simulations):
    print(f"\n========== Run {i_sim+1}/{num_simulations} ==========")
    sim_failed = False
    # =================================================
    # SGTS-LHS Sampling
    # =================================================
    sgts = SGTS_LHS(
        n_dim=n_dim,
        param_names=param_names,
        x_limits_global=xlimits_global,
        objective_func=lambda x: f(*x),
        # Sampling budget
        n_samples_s1_factor=factor_s1,
        n_samples_s2_factor=factor_s2,
        # SHAP-ROI related hyperparameters
        k_best_for_roi_bounds_factor=k_best_factor,
        roi_padding_factor_important=roi_padding_factor_imp,
        roi_padding_factor_non_important=roi_padding_factor_nonimp,
        # Other optional settings
        preliminary_surrogate_model_type='rf',
        n_top_shap_features=n_sgts_important_features,
        # Random seed (to ensure each simulation is different)
        s1_lhs_seed=i_sim,
        preliminary_model_train_seed=1000 + i_sim,
        s2_lhs_roi_seed=10000 + i_sim
    )
    (X_train_sgts, Y_train_sgts, total_calls_sgts, shap_plot_details) = sgts.generate_samples()
    # =================================================
    # LHS Sampling (simply set the second-stage sampling factor to 0)
    # =================================================
    sgts_single = SGTS_LHS(
        n_dim=n_dim,
        param_names=param_names,
        x_limits_global=xlimits_global,
        objective_func=lambda x: f(*x),
        n_samples_s1_factor=n_samples_total // n_dim,
        n_samples_s2_factor=0,          # <- Single-stage
        k_best_for_roi_bounds_factor=0, # ↓ These three can be any value
        roi_padding_factor_important=0,
        roi_padding_factor_non_important=0,
        s1_lhs_seed=20000 + i_sim
    )
    X_train_1stage, Y_train_1stage, _, _ = sgts_single.generate_samples()

    # =================================================
    # Train the final surrogate models for both sampling methods using Kriging
    # =================================================
    krig_sgts = KRG(print_global=False, theta0=[1e-2]*n_dim)
    krig_1stage  = KRG(print_global=False, theta0=[1e-2]*n_dim)

    krig_sgts.set_training_values(X_train_sgts, Y_train_sgts)
    krig_sgts.train()
    krig_1stage.set_training_values(X_train_1stage, Y_train_1stage)
    krig_1stage.train()

    # =================================================
    # Use GA to find the minimum of the final surrogate models
    # =================================================
    def make_obj(krig_model):
        def obj(p):
            return krig_model.predict_values(np.array(p).reshape(1, -1)).item()
        return obj

    ga_sgts = GA(
        func=make_obj(krig_sgts),
        n_dim=n_dim,
        size_pop=ga_pop_size,
        max_iter=ga_max_iter,
        lb=lb_global.tolist(),
        ub=ub_global.tolist(),
        precision=1e-4,
        prob_mut=0.02
    )
    best_p_sgts, best_val_pred_list = ga_sgts.run()
    best_val_pred_sgts = best_val_pred_list[0]
    true_val_best_sgts = f(*best_p_sgts)

    ga_1stage = GA(
        func=make_obj(krig_1stage),
        n_dim=n_dim,
        size_pop=ga_pop_size,
        max_iter=ga_max_iter,
        lb=lb_global.tolist(),
        ub=ub_global.tolist(),
        precision=1e-4,
        prob_mut=0.02
    )
    best_p_1stage, best_val_pred_1stage_list = ga_1stage.run()
    best_val_pred_1stage = best_val_pred_1stage_list[0]
    true_val_best_1stage = f(*best_p_1stage)

    # =================================================
    # Record Results
    # =================================================
    """
    'best_params': Best parameters found in each simulation (50, 2)
    'pred_vals_at_best': Predicted values from the surrogate model at the best parameter points (50,)
    'true_vals_at_best': True function values at the best parameter points (most critical) (50,)
    'rmse': Global RMSE of the surrogate model (50,)
    'ga_histories': Convergence history of the genetic algorithm (50, 100)
    """
    # ! SGTS-LHS Results
    results_sgts['best_params'].append(best_p_sgts)
    results_sgts['pred_vals_at_best'].append(best_val_pred_sgts)
    results_sgts['true_vals_at_best'].append(true_val_best_sgts)
    results_sgts['ga_histories'].append([min(hist) for hist in ga_sgts.all_history_Y] if not sim_failed else [])

    Y_pred_test = krig_sgts.predict_values(X_test_fixed)
    rmse_sgts = np.sqrt(mean_squared_error(Y_test_true_fix, Y_pred_test))
    results_sgts['rmse'].append(rmse_sgts)
    
    # ! LHS Results    
    results_1stage['best_params'].append(best_p_1stage)
    results_1stage['pred_vals_at_best'].append(best_val_pred_1stage)
    results_1stage['true_vals_at_best'].append(true_val_best_1stage)
    results_1stage['ga_histories'].append([min(hist) for hist in ga_1stage.all_history_Y] if not sim_failed else [])

    Y_pred_test_1stage = krig_1stage.predict_values(X_test_fixed)
    rmse_1stage = np.sqrt(mean_squared_error(Y_test_true_fix, Y_pred_test_1stage))

    results_1stage['rmse'].append(rmse_1stage)
    
    last_sim_failed = sim_failed

# =================================================
# Results Summary and Comparison
# =================================================
print("\n\n--- Summary of Multiple Simulation Runs ---")
print(f"Total number of simulations: {num_simulations}")
print(f"Total sample points per simulation (number of true function calls): {n_samples_total}")

valid_true_vals_sgts = [v for v in results_sgts['true_vals_at_best'] if not np.isnan(v)]
valid_rmse_sgts = [v for v in results_sgts['rmse'] if not np.isnan(v)]
valid_true_vals_1stage = [v for v in results_1stage['true_vals_at_best'] if not np.isnan(v)]
valid_rmse_1stage = [v for v in results_1stage['rmse'] if not np.isnan(v)]

print("\n--- SHAP-Guided Two-Stage Strategy (Statistical Results) ---")
if valid_true_vals_sgts:
    print(f"  True function value at the optimum found by GA ({len(valid_true_vals_sgts)} valid simulations):")
    print(f"    Mean: {np.mean(valid_true_vals_sgts):.4f}")
    print(f"    Standard Deviation: {np.std(valid_true_vals_sgts):.4f}")
    print(f"    Minimum (best performance): {np.min(valid_true_vals_sgts):.4f}")
    print(f"    Median: {np.median(valid_true_vals_sgts):.4f}")

if valid_rmse_sgts:
    print(f"  Global accuracy of the surrogate model (RMSE) ({len(valid_rmse_sgts)} valid simulations):")
    print(f"    Mean: {np.mean(valid_rmse_sgts):.4f}")
    print(f"    Standard Deviation: {np.std(valid_rmse_sgts):.4f}")


print("\n--- Single-Stage Strategy (Statistical Results) ---")
if valid_true_vals_1stage:
    print(f"  True function value at the optimum found by GA ({len(valid_true_vals_1stage)} valid simulations):")
    print(f"    Mean: {np.mean(valid_true_vals_1stage):.4f}")
    print(f"    Standard Deviation: {np.std(valid_true_vals_1stage):.4f}")
    print(f"    Minimum (best performance): {np.min(valid_true_vals_1stage):.4f}")
    print(f"    Median: {np.median(valid_true_vals_1stage):.4f}")

if valid_rmse_1stage:
    print(f"  Global accuracy of the surrogate model (RMSE) ({len(valid_rmse_1stage)} valid simulations):")
    print(f"    Mean: {np.mean(valid_rmse_1stage):.4f}")
    print(f"    Standard Deviation: {np.std(valid_rmse_1stage):.4f}")

# =================================================
# Plot 1
# =================================================
nx, ny = 300, 300
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

Z = f(X, Y)

x = np.linspace(0, 1, Z.shape[1])
y = np.linspace(0, 1, Z.shape[0])
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(7, 4))
plt.imshow(Z, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap="jet", alpha=0.8)
plt.colorbar(label='f(x, y)')
contours = plt.contour(X, Y, Z, colors="black", linewidths=0.8, levels=8, linestyles='dashed')
plt.clabel(contours, inline=True, fontsize=10, fmt="%.2f", colors="black",)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# =================================================
# Plot 2
# =================================================
xs = np.linspace(lb_global[0], ub_global[0], 100)
ys = np.linspace(lb_global[1], ub_global[1], 100)
X, Y = np.meshgrid(xs, ys)
Z_true = f(X, Y)

preds = {}
for key, model in (('SHAP', krig_sgts), ('1-Stage', krig_1stage)):
    try:
        Zp = model.predict_values(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    except:
        Zp = np.zeros_like(X)
    preds[key] = Zp

vmin_row1 = min(Z_true.min(), preds['1-Stage'].min())
vmax_row1 = max(Z_true.max(), preds['1-Stage'].max())
vmin_row2 = min(Z_true.min(), preds['SHAP'].min())
vmax_row2 = max(Z_true.max(), preds['SHAP'].max())

fig, axes = plt.subplots(2, 3, figsize=(20, 6))
axes = axes.flat

tr = X_train_sgts if key=='SHAP' else X_train_1stage
res = results_sgts if key=='SHAP' else results_1stage

sgts_points = np.array(results_sgts['best_params'])
lhs_points = np.array(results_1stage['best_params'])

ax = axes[0]
cp1 = ax.contourf(X, Y, Z_true, levels=50, cmap='jet', vmin=vmin_row1, vmax=vmax_row1, alpha=0.8)
ax.scatter(tr[:,0], tr[:,1], c="yellow", s=15, edgecolor='black')

ax = axes[1]
cp = ax.contourf(X, Y, preds[key], levels=50, cmap='jet', vmin=vmin_row1, vmax=vmax_row1, alpha=0.8)
fig.colorbar(cp1, ax=[axes[0], axes[1]]) # Create a shared colorbar for the first two subplots in the row
cp = ax.contour(X, Y, preds[key], colors="black", linewidths=0.8, levels=5, linestyles='dashed')
ax.clabel(cp, inline=True, fontsize=10, fmt="%.2f", colors="black",)
ax.scatter(lhs_points[:, 0], lhs_points[:, 1], c='yellow', marker='x', s=20, label='LHS Optima')

ax = axes[2]
err = np.abs(Z_true - preds['1-Stage'])
cp = ax.contourf(X, Y, err, levels=50, cmap='Reds')
plt.colorbar(cp, ax=ax)
cp = ax.contour(X, Y, err, colors="black", linewidths=0.8, levels=5, linestyles='dashed')
ax.clabel(cp, inline=True, fontsize=10, fmt="%.2f", colors="black",)
bp = res['best_params'][2]

ax = axes[3]
n_s1 = n_samples_stage1 
X_stage1 = X_train_sgts[:n_s1, :]
Y_stage1 = Y_train_sgts[:n_s1]
X_stage2 = X_train_sgts[n_s1:, :]
Y_stage2 = Y_train_sgts[n_s1:]
cp = ax.contourf(X, Y, Z_true, levels=50, cmap='jet', vmin=vmin_row2, vmax=vmax_row2, alpha=0.8)
ax.scatter(X_stage1[:,0], X_stage1[:,1], c="yellow", marker='s', s=15, edgecolor='black', alpha=0.8)
ax.scatter(X_stage2[:, 0], X_stage2[:, 1], c='red', marker='s', edgecolor='black', s=15, label='LHS Optima')

ax = axes[4]
cp2 = ax.contourf(X, Y, preds['SHAP'], levels=50, cmap='jet', vmin=vmin_row2, vmax=vmax_row2)
fig.colorbar(cp2, ax=[axes[3], axes[4]]) # Create a shared colorbar for the second two subplots in the row
cp = ax.contour(X, Y, preds['SHAP'], colors="black", linewidths=0.8, levels=5, linestyles='dashed')
ax.clabel(cp, inline=True, fontsize=10, fmt="%.2f", colors="black",)
ax.scatter(sgts_points[:, 0], sgts_points[:, 1], c='red', marker='+', s=20, label='LHS Optima')

ax = axes[5]
err = np.abs(Z_true - preds['SHAP'])
cp = ax.contourf(X, Y, err, levels=50, cmap='Reds')
plt.colorbar(cp, ax=ax)
cp = ax.contour(X, Y, err, colors="black", linewidths=0.8, levels=5, linestyles='dashed')
ax.clabel(cp, inline=True, fontsize=10, fmt="%.2f", colors="black",)
bp = res['best_params'][2]
plt.show()

# =================================================
# Plot 3
# =================================================
df_rmse = pd.DataFrame({
    'RMSE': results_sgts['rmse'] + results_1stage['rmse'],
    'Method': ['SGTS-LHS'] * len(results_sgts['rmse']) + ['LHS'] * len(results_1stage['rmse'])
})
df_pred_vals = pd.DataFrame({
    'True Value at Optimum': results_sgts['pred_vals_at_best'] + results_1stage['pred_vals_at_best'],
    'Method': ['SGTS-LHS'] * len(results_sgts['pred_vals_at_best']) + ['LHS'] * len(results_1stage['pred_vals_at_best'])
})
df_true_vals = pd.DataFrame({
    'True Value at Optimum': results_sgts['true_vals_at_best'] + results_1stage['true_vals_at_best'],
    'Method': ['SGTS-LHS'] * len(results_sgts['true_vals_at_best']) + ['LHS'] * len(results_1stage['true_vals_at_best'])
})

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
custom_palette = ['#EB7E60', '#1f77b4', '#2ca02c']
palette = custom_palette[:len(df_true_vals['Method'].unique())]

sns.boxplot(
    data=df_rmse,
    x='Method',
    y='RMSE',
    ax=axes[0],
    showfliers=False,
    boxprops={'edgecolor': 'black', 'alpha': 0.8, 'facecolor':'None'},
    whiskerprops={'color': 'black', 'alpha': 0.8},
    capprops={'color': 'black', 'alpha': 0.8},
    medianprops={'color': 'red', 'linewidth': 1.2},
    width=0.3,
)
sns.stripplot(
    data=df_rmse,
    x='Method', 
    y='RMSE',
    ax=axes[0],
    color=".25",
    alpha=0.6,
    size=5,
    jitter=0.05,
    palette=palette
)
axes[0].grid(visible=True, which='major', axis='both', alpha=0.5, linestyle='--')
axes[0].set_axisbelow(True)

sns.boxplot(
    data=df_pred_vals,
    x='Method',
    y='True Value at Optimum',
    ax=axes[1],
    showfliers=False,
    boxprops={'edgecolor': 'black', 'alpha': 0.8, 'facecolor':'None'},
    whiskerprops={'color': 'black', 'alpha': 0.8},
    capprops={'color': 'black', 'alpha': 0.8},
    medianprops={'color': 'red', 'linewidth': 1.2},
    width=0.3, # Control the width of the entire group
)
sns.stripplot(
    data=df_pred_vals,
    x='Method', 
    y='True Value at Optimum',
    ax=axes[1],
    color=".25",
    alpha=0.6,
    size=5,
    jitter=0.05,
    palette=palette
)
axes[1].grid(visible=True, which='major', axis='both', alpha=0.5, linestyle='--')
axes[1].set_axisbelow(True)

sns.boxplot(
    data=df_true_vals,
    x='Method',
    y='True Value at Optimum',
    ax=axes[2],
    showfliers=False,
    boxprops={'edgecolor': 'black', 'alpha': 0.8, 'facecolor':'None'},
    whiskerprops={'color': 'black', 'alpha': 0.8},
    capprops={'color': 'black', 'alpha': 0.8},
    medianprops={'color': 'red', 'linewidth': 1.2},
    width=0.3, # Control the width of the entire group
)
sns.stripplot(
    data=df_true_vals,
    x='Method', 
    y='True Value at Optimum',
    ax=axes[2],
    color=".25",
    alpha=0.6,
    size=5,
    jitter=0.05,
    palette=palette
)
axes[2].grid(visible=True, which='major', axis='both', alpha=0.5, linestyle='--')
axes[2].set_axisbelow(True)

# Display the plot
plt.tight_layout()
plt.show()

# =================================================
# Plot 4
# =================================================
data_ga   = np.array(results_1stage['ga_histories'])
data_sgts = np.array(results_sgts['ga_histories'])
x = np.arange(data_ga.shape[1])

# Calculate Standard Error of the Mean (SEM)
mean_ga   = data_ga.mean(0)
sem_ga    = data_ga.std(0) / np.sqrt(data_ga.shape[0])
mean_s    = data_sgts.mean(0)
sem_s     = data_sgts.std(0) / np.sqrt(data_sgts.shape[0])

# Smoothing
window = 7
mg = pd.Series(mean_ga).rolling(window, center=True, min_periods=1).mean()
lg = pd.Series(mean_ga - sem_ga).rolling(window, center=True, min_periods=1).mean()
ug = pd.Series(mean_ga + sem_ga).rolling(window, center=True, min_periods=1).mean()

ms = pd.Series(mean_s).rolling(window, center=True, min_periods=1).mean()
ls = pd.Series(mean_s - sem_s).rolling(window, center=True, min_periods=1).mean()
us = pd.Series(mean_s + sem_s).rolling(window, center=True, min_periods=1).mean()

plt.figure(figsize=(16,2.5))

plt.plot(
    x, mg,
    label='GA (smoothed)',
    linewidth=2,
    marker='o',
    markevery=5,
    linestyle='--',
    markeredgecolor='black',
    color='#4e9381'
)
plt.fill_between(
    x, lg, ug,
    alpha=0.15,
    facecolor='#4e9381',
    label='GA ± SEM'
)

# SGTS-LHS line and fill
plt.plot(
    x, ms,
    label='SGTS-LHS (smoothed)',
    linewidth=2,
    marker='o',
    markevery=5,
    linestyle='--',
    markeredgecolor='black',
    color='#b86366'
)
plt.fill_between(
    x, ls, us,
    alpha=0.15,
    facecolor='#b86366',
    label='SGTS-LHS ± SEM'
)

plt.xlabel('Epoch')
plt.ylabel('f(x, y)')
plt.legend()
plt.grid(visible=True, which='major', axis='both', alpha=0.5, linestyle='--')
plt.show()