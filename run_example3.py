import os 
import time
import pickle
import random
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from sko.GA import GA

from smt.sampling_methods import LHS
from smt.surrogate_models import KRG

# ========== Custom Modules ==========
from models.sampler import SGTS_LHS
from utils.modflow_model import modflow_model

# ========== Global Constants and True Values ==========
data_path = "./data/KLE.npy"
assert os.path.exists(data_path), f"{data_path} does not exist, please check the path"
base_seed = 1234

(nx_loaded,
 ny_loaded,
 mean_logk_loaded,
 lamda_xy_loaded,
 fn_x_loaded,
 fn_y_loaded,
 true_kesi) = np.load(data_path, allow_pickle=True).tolist()

# ========== Define Global Parameters for Multiprocessing ==========
n_dim = true_kesi.shape[1]

lb = -2.5 * np.ones(n_dim)  # Lower bounds
ub =  2.5 * np.ones(n_dim)  # Upper bounds

n_samples_s1_factor  = 300 # Factor for stage-1 samples 300
n_samples_s2_factor  = 30  # Factor for stage-2 samples 30

ga_kwargs = {
    "n_dim": n_dim,
    "size_pop": 500,   # Population size
    "max_iter": 200,   # Maximum iterations
    "prob_mut": 0.02,  # Mutation probability
    "lb": lb.tolist(), # Lower bounds
    "ub": ub.tolist(), # Upper bounds
}

# ========== Objective Function ==========
def calculate_logk(nx, ny, mean_logk, lamda_xy, fn_x, fn_y, kesi):
    """Reconstruct the log(k) field based on the parameter kesi"""
    kesi = kesi.reshape(1, -1)
    logk = np.zeros((nx, ny))
    for ix in range(nx):
        for iy in range(ny):
            logk[iy, ix] = mean_logk + np.sum(np.sqrt(lamda_xy) * fn_x[ix][0] * fn_y[iy][0] * kesi.transpose())
    return logk
    
def forward_model(kesi):
    """Forward model: kesi -> concentration or head observations based on MODFLOW6)"""
    logk = calculate_logk(nx_loaded, ny_loaded,
                          mean_logk_loaded, lamda_xy_loaded,
                          fn_x_loaded, fn_y_loaded,
                          np.asarray(kesi))
    k = np.exp(logk).reshape(1, nx_loaded, ny_loaded)
    
    head, conc = modflow_model(k=k)
    conc = conc.reshape(10, nx_loaded, ny_loaded)
    head = head.reshape(10, nx_loaded, ny_loaded)
    return np.array([head, conc])
    # return k

# True observations (Flatten true observations)
true_obs_flat = forward_model(true_kesi).ravel()

def objective_function(kesi):
    """MSE between the true and simulated values"""
    pred = forward_model(kesi).ravel()
    return mean_squared_error(true_obs_flat, pred)

# ========== Worker Function for Multiprocessing ==========
def worker_function(run_info):
    """Worker function for multiprocessing: performs one run per process"""
    run_number, total_runs = run_info  # Unpack run_info
    print(f"\n================== Starting Run {run_number+1}/{total_runs} (Process ID: {os.getpid()}) ==================")
    seed_base = base_seed + run_number
    np.random.seed(seed_base)
    random.seed(seed_base)

    # ! ========== Traditional Single-stage LHS Sampling ==========
    lhs_sampler = LHS(xlimits=np.vstack((lb, ub)).T, random_state=seed_base)
    X_lhs = lhs_sampler(n_dim * (n_samples_s1_factor + n_samples_s2_factor))
    
    # Evaluate objective function for LHS samples
    y_lhs_list = []
    for i_sample, x_sample in enumerate(X_lhs):
        y_lhs_list.append(objective_function(x_sample))
    y_lhs = np.array(y_lhs_list)

    print(f"Process ID: {os.getpid()} Single-stage sampling completed")

    rf_ga = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=seed_base,
        n_jobs=1,
    )
    rf_ga.fit(X_lhs, y_lhs)

    # Alternative: use KRG surrogate model
    # rf_ga = KRG(print_global=False, nugget=1e-6)
    # rf_ga.set_training_values(X_lhs, y_lhs)
    # rf_ga.train()

    def f_surrogate_ga(x):
        """Surrogate function for GA: predict using RF"""
        x = np.asarray(x).reshape(1, -1)
        return rf_ga.predict(x).item()

    current_ga_kwargs = ga_kwargs.copy()
    ga_solver_ga = GA(func=f_surrogate_ga, **current_ga_kwargs)
    best_kesi_ga, _ = ga_solver_ga.run(max_iter=current_ga_kwargs['max_iter'])  # Ensure max_iter is passed if run overrides defaults
    best_kesi_ga = np.asarray(best_kesi_ga)

    # ! ========== SGTS-LHS Sampling ==========

    sgts_prelim_model_params = {'n_estimators': 150, 
                                'max_depth': 15, 
                                'min_samples_split': 5, 
                                'min_samples_leaf': 2,
                                'n_jobs': 1
                               }

    sgts_sampler = SGTS_LHS(
        n_dim=n_dim,
        param_names=[f'kesi_{i+1}' for i in range(n_dim)],
        x_limits_global=np.vstack((lb, ub)).T,
        objective_func=objective_function,
        n_samples_s1_factor=n_samples_s1_factor,
        n_samples_s2_factor=n_samples_s2_factor,
        k_best_for_roi_bounds_factor=0.01,
        roi_padding_factor_important=0.01,
        roi_padding_factor_non_important=0.02,
        preliminary_surrogate_model_type="rf",
        preliminary_surrogate_model_params=sgts_prelim_model_params,
        n_top_shap_features=5,
        s1_lhs_seed=seed_base+11,
        preliminary_model_train_seed=seed_base+22,
        s2_lhs_roi_seed=seed_base+33
    )
    X_sgts, y_sgts, _, _ = sgts_sampler.generate_samples()
    
    print(f"Run {run_number+1}, SGTS-LHS, Finished generate_samples.")

    rf_sgts = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=seed_base,
        n_jobs=1,
    )
    rf_sgts.fit(X_sgts, y_sgts.ravel())

    # Alternative
    # rf_sgts = KRG(print_global=False, nugget=1e-6)
    # rf_sgts.set_training_values(X_sgts,y_sgts)
    # rf_sgts.train()

    def f_surrogate_sgts(x):
        """Surrogate function for SGTS-LHS"""
        x = np.asarray(x).reshape(1, -1)
        return rf_sgts.predict(x).item()

    ga_solver_sgts = GA(func=f_surrogate_sgts, **current_ga_kwargs)
    best_kesi_sgts, _ = ga_solver_sgts.run(max_iter=current_ga_kwargs['max_iter'])
    best_kesi_sgts = np.asarray(best_kesi_sgts)
    
    # ! ========== Evaluation ==========

    pred_ga   = forward_model(best_kesi_ga  ).reshape(-1)
    pred_sgts = forward_model(best_kesi_sgts).reshape(-1)

    run_metrics_ga_obs = [
        mean_squared_error(true_obs_flat.ravel(), pred_ga.ravel()),
        mean_absolute_error(true_obs_flat.ravel(), pred_ga.ravel()),
        r2_score(true_obs_flat.ravel(), pred_ga.ravel()),
    ]

    run_metrics_sgts_obs = [
        mean_squared_error(true_obs_flat.ravel(), pred_sgts.ravel()),
        mean_absolute_error(true_obs_flat.ravel(), pred_sgts.ravel()),
        r2_score(true_obs_flat.ravel(), pred_sgts.ravel()),
    ]

    run_metrics_ga_par = [
        mean_squared_error(true_kesi.ravel(), best_kesi_ga.ravel()),
        mean_absolute_error(true_kesi.ravel(), best_kesi_ga.ravel()),
        r2_score(true_kesi.ravel(), best_kesi_ga.ravel()),
    ]

    run_metrics_sgts_par = [
        mean_squared_error(true_kesi.ravel(), best_kesi_sgts.ravel()),
        mean_absolute_error(true_kesi.ravel(), best_kesi_sgts.ravel()),
        r2_score(true_kesi.ravel(), best_kesi_sgts.ravel()),
    ]

    run_perm_ga   = forward_model(best_kesi_ga  )[0]
    run_perm_sgts = forward_model(best_kesi_sgts)[0]
    
    print(f"========== Finished Run {run_number+1}/{total_runs} (Process ID: {os.getpid()}) ==========")
    return (
        run_metrics_ga_obs,
        run_metrics_sgts_obs,
        run_metrics_ga_par,
        run_metrics_sgts_par,
        best_kesi_ga,
        best_kesi_sgts,
        run_perm_ga,
        run_perm_sgts,
    )

def to_arr(lst):
    """Convert list to NumPy array"""
    return np.asarray(lst)

if __name__ == '__main__':
    # ========== Hyperparameters ==========
    n_runs = 50

    # ========== Initialize Containers for Main Process ==========
    metrics_ga_obs_all = []
    metrics_sgts_obs_all = []
    metrics_ga_par_all = []
    metrics_sgts_par_all = []
    best_kesi_ga_all = []
    best_kesi_sgts_all = []
    perm_ga_list_all = []
    perm_sgts_list_all = []

    # Determine number of processes based on CPU cores
    num_processes = 6
    num_processes = min(n_runs, num_processes)
    cpu_cores = multiprocessing.cpu_count()

    print(f"Starting multiprocessing with {num_processes} processes for {n_runs} runs...")
    
    start_time_main = time.time()  # Record start time

    # Prepare arguments for worker_function: a list of (run_index, total_runs) tuples
    worker_args = [(i, n_runs) for i in range(n_runs)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Execute worker_function in parallel and collect results
        results = pool.map(worker_function, worker_args)
    
    end_time_main = time.time()

    print(f"\nAll {n_runs} runs completed in {end_time_main - start_time_main:.2f} seconds using {num_processes} processes.")

    # Unpack results from the list of tuples returned by pool.map
    for res_tuple in results:
        if res_tuple:  # Check if the result is not None
            metrics_ga_obs_all.append(res_tuple[0])
            metrics_sgts_obs_all.append(res_tuple[1])
            metrics_ga_par_all.append(res_tuple[2])
            metrics_sgts_par_all.append(res_tuple[3])
            best_kesi_ga_all.append(res_tuple[4])
            best_kesi_sgts_all.append(res_tuple[5])
            perm_ga_list_all.append(res_tuple[6])
            perm_sgts_list_all.append(res_tuple[7])
        else:
            print("Warning: A worker process might have failed and returned None.")

    # ========== Statistics ==========
    ga_obs   = to_arr(metrics_ga_obs_all)
    sgts_obs = to_arr(metrics_sgts_obs_all)
    ga_par   = to_arr(metrics_ga_par_all)
    sgts_par = to_arr(metrics_sgts_par_all)

    # Make perm_lists available for plotting
    perm_ga_list = perm_ga_list_all
    perm_sgts_list = perm_sgts_list_all

    def mean_std(mat, idx):
        """Compute mean and std for a given metric matrix"""
        if mat.shape[0] == 0:  # Handle case with no results
            return np.nan, np.nan
        return mat[:, idx].mean(), mat[:, idx].std()

    names = ["MSE", "MAE", "R2"]
    print("\n================= Mean ± Std =================")
    for i, name in enumerate(names):
        m, s = mean_std(ga_obs, i)
        print(f"Obs-GA   {name}: {m:.4f} ± {s:.4f}")
        m, s = mean_std(sgts_obs, i)
        print(f"Obs-SGTS {name}: {m:.4f} ± {s:.4f}")
        print("----")
    for i, name in enumerate(names):
        m, s = mean_std(ga_par, i)
        print(f"Par-GA   {name}: {m:.4f} ± {s:.4f}")
        m, s = mean_std(sgts_par, i)
        print(f"Par-SGTS {name}: {m:.4f} ± {s:.4f}")
        print("----")

    # ========== Save Results ==========
    results_to_save = {

        "metrics_ga_obs": ga_obs,
        "metrics_sgts_obs": sgts_obs,
        "metrics_ga_par": ga_par,
        "metrics_sgts_par": sgts_par,

        "best_kesi_ga_runs": best_kesi_ga_all,
        "best_kesi_sgts_runs": best_kesi_sgts_all,
        "perm_ga_runs": perm_ga_list,
        "perm_sgts_runs": perm_sgts_list,
        
        "true_parameters": true_kesi,
        "true_observations": true_obs_flat,
        # "true_permeability_field": true_perm_field,
        
        "n_runs": n_runs,
        "dimensions": {
            "nx": nx_loaded,
            "ny": ny_loaded
        },
        "hyperparameters": {
            "n_samples_s1_factor": n_samples_s1_factor,
            "n_samples_s2_factor": n_samples_s2_factor,
            "ga_kwargs": ga_kwargs
        }
    }
    output_filename = f"experiment_results_runs_{n_runs}.pkl"
    
    try:
        with open(output_filename, 'wb') as f:
            pickle.dump(results_to_save, f)
        print(f"Results successfully saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving results: {e}")