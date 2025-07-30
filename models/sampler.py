import numpy as np
import shap
from smt.sampling_methods import LHS
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

class SGTS_LHS:
    def __init__(self,
                 n_dim: int,                           # Parameter dimensionality
                 param_names: list,                     # List of parameter names
                 x_limits_global: np.ndarray,           # Global search range for parameters, shape(n_dim, 2)
                 objective_func: callable,              # Objective function

                 n_samples_s1_factor: int,              # Stage 1 sampling factor
                 n_samples_s2_factor: int,              # Stage 2 sampling factor

                 k_best_for_roi_bounds_factor: float,   # Proportion of best samples used to determine ROI bounds
                 roi_padding_factor_important: float,   # ROI boundary padding factor for important features
                 roi_padding_factor_non_important: float, # ROI boundary padding factor for non-important features

                 preliminary_surrogate_model_type: str = 'rf', # Type of the preliminary surrogate model
                 preliminary_surrogate_model_params: dict = None, # Parameters for the preliminary surrogate model
                 n_top_shap_features: int = 3,          # Number of important features from SHAP analysis

                 s1_lhs_seed: int = None,               # Random seed for Stage 1 LHS
                 preliminary_model_train_seed: int = None, # Random seed for training the preliminary model
                 s2_lhs_roi_seed: int = None            # Random seed for Stage 2 LHS in ROI
                ):
        
        # Basic parameters
        self.n_dim = n_dim
        self.param_names = param_names
        self.x_limits_global = x_limits_global
        self.objective_func = objective_func

        # Sampling parameters
        self.n_samples_s1_factor = n_samples_s1_factor
        self.n_samples_s2_factor = n_samples_s2_factor

        # ROI parameters
        self.k_best_factor = k_best_for_roi_bounds_factor
        self.roi_padding_important = roi_padding_factor_important
        self.roi_padding_non_important = roi_padding_factor_non_important

        # Surrogate model parameters
        self.model_type = preliminary_surrogate_model_type
        self.model_params = preliminary_surrogate_model_params if preliminary_surrogate_model_params is not None else {}
        self.n_top_features = min(n_top_shap_features, n_dim)  # Ensure it does not exceed the total dimensionality

        # Random seeds
        self.s1_seed = s1_lhs_seed
        self.model_seed = preliminary_model_train_seed
        self.s2_seed = s2_lhs_roi_seed
        
        # Parameter validation
        if not isinstance(self.param_names, list) or len(self.param_names) != self.n_dim:
            raise ValueError("param_names must be a list of length n_dim")
        if self.x_limits_global.shape != (self.n_dim, 2):
            raise ValueError("x_limits_global must be a numpy array of shape (n_dim, 2)")


    def _evaluate_objective_batch(self, x_data: np.ndarray) -> np.ndarray:
        """Batch evaluate the objective function and ensure the output is a 2D array (n_samples, n_outputs)"""
        y_list = []
        for i in range(x_data.shape[0]):
            try:
                y_val = self.objective_func(x_data[i])
                y_list.append(y_val)
            except Exception as e:
                print(f"Error evaluating sample {i} (value: {x_data[i]}): {e}")
                # Raise exception directly on error
                raise

        # Convert to numpy array and ensure it is float type
        y_array = np.array(y_list, dtype=float)
        
        # Handle different output dimensions
        if y_array.ndim == 0:  # If a single scalar is returned
            y_array = y_array.reshape(1, 1)
        elif y_array.ndim == 1:  # If a 1D array is returned
            y_array = y_array.reshape(-1, 1)
            
        # Validate output length
        if x_data.shape[0] > 0 and y_array.shape[0] != x_data.shape[0]:
            raise ValueError(f"Objective function output length mismatch. Expected {x_data.shape[0]}, got {y_array.shape[0]}")
        return y_array

    def _perform_single_stage_lhs(self):
        """Perform single-stage LHS sampling"""
        # Calculate the total number of sample points
        total_samples = self.n_samples_s1_factor * self.n_dim
        print(f"\n--- SGTS_LHS: Executing single-stage LHS strategy ({total_samples} points) ---")

        # Create LHS sampler and generate samples
        lhs_obj = LHS(xlimits=self.x_limits_global, criterion='m', random_state=self.s1_seed)
        x_train = lhs_obj(total_samples)
        y_train = self._evaluate_objective_batch(x_train)
        total_calls = x_train.shape[0]

        # Output information
        if y_train.shape[1] > 1:
            print(f"  Objective function detected {y_train.shape[1]} outputs")
        print(f"  Single-stage sampling complete. Total function calls: {total_calls}")
        return x_train, y_train, total_calls, None

    def _perform_stage1_sampling_and_eval(self, n_s1: int):
        """Perform Stage 1 LHS sampling and evaluate the objective function"""
        print(f"  Stage 1: Global LHS ({n_s1} points)...")
        
        # Create LHS sampler and generate samples
        lhs_s1 = LHS(xlimits=self.x_limits_global, criterion='m', random_state=self.s1_seed)
        x_s1 = lhs_s1(n_s1)
        y_s1 = self._evaluate_objective_batch(x_s1)

        # Output information
        if y_s1.shape[1] > 1:
            print(f"  Objective function detected {y_s1.shape[1]} outputs in Stage 1")
        return x_s1, y_s1

    def _train_preliminary_surrogate(self, x_s1: np.ndarray, y_s1: np.ndarray):
        """Train the preliminary surrogate model for SHAP analysis"""
        print("  Training preliminary surrogate model for SHAP analysis...")
        
        # Copy model parameters and set random seed
        model_params = self.model_params.copy()
        model_params['random_state'] = self.model_seed

        # ! Random Forest model
        if self.model_type == 'rf':
            # Set default parameters
            if 'n_estimators' not in model_params:
                model_params['n_estimators'] = 100
            if 'n_jobs' not in model_params:
                model_params['n_jobs'] = -1

            # Create and train the model
            model = RandomForestRegressor(**model_params)
            
            # Handle single-output and multi-output cases
            y_fit = y_s1
            if y_s1.shape[1] == 1:  # Single-output case
                y_fit = y_s1.ravel()  # For single-output RF, a 1D y is usually expected

            model.fit(x_s1, y_fit)
            print(f"  Preliminary Random Forest model trained")
            

            # --- Add the following evaluation code ---
            print("  --- Diagnostics: Evaluating preliminary surrogate model performance on S1 training set ---")
            try:
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                import numpy as np

                y_s1_pred_on_train = model.predict(x_s1)

                # Ensure y_fit and y_s1_pred_on_train have compatible shapes for comparison
                # y_fit is already in the correct shape for training (usually 1D)
                # y_s1_pred_on_train is also usually 1D

                r2_on_train = r2_score(y_fit, y_s1_pred_on_train)
                mse_on_train = mean_squared_error(y_fit, y_s1_pred_on_train)
                rmse_on_train = np.sqrt(mse_on_train)
                mae_on_train = mean_absolute_error(y_fit, y_s1_pred_on_train)

                print(f"    R2 Score (training set): {r2_on_train:.4f}")
                print(f"    Root Mean Squared Error (RMSE, training set): {rmse_on_train:.4f}")
                print(f"    Mean Absolute Error (MAE, training set): {mae_on_train:.4f}")

                # Compare MAE with the standard deviation of y_fit to see how much better the model is than a simple mean prediction
                y_fit_std = np.std(y_fit)
                print(f"    Standard deviation of target y_fit: {y_fit_std:.4f}")
                if y_fit_std > 1e-9: # Avoid division by zero
                    print(f"    MAE / std(y_fit): {mae_on_train / y_fit_std:.4f} (smaller is better, ideally much less than 1)")
                
                if r2_on_train < 0.1: # A low threshold, can be adjusted depending on the problem
                    print(f"    Warning: The R2 score ({r2_on_train:.4f}) of the preliminary surrogate model on the training set is very low, which may indicate that the model failed to learn the data effectively.")
                elif r2_on_train < 0.5:
                     print(f"    Note: The R2 score ({r2_on_train:.4f}) of the preliminary surrogate model on the training set is moderate. The reliability of SHAP results may be affected.")
                else:
                    print(f"    The R2 score ({r2_on_train:.4f}) of the preliminary surrogate model on the training set is acceptable/good.")

            except Exception as e:
                print(f"    Error during evaluation of the preliminary model: {e}")
            print("  --- End of Diagnostics ---")
            # --- End of evaluation code ---
            
            # Output multi-output information
            if y_s1.shape[1] > 1 and y_fit.ndim > 1 and y_fit.shape[1] > 1:
                print(f"    Model trained for {y_s1.shape[1]} outputs")
            return model
        
        # ! Kriging surrogate model
        if self.model_type == 'kriging':
            # Import SMT Kriging model
            try:
                from smt.surrogate_models import KRG
            except ImportError as e:
                raise ImportError("Please install the smt library first: pip install smt") from e
            
            # Create and train Kriging model
            model = KRG(**model_params)
            model.set_training_values(x_s1, y_s1)
            model.train()

            print(f"  Preliminary Kriging surrogate model trained")

            # --- Add the following evaluation code ---
            print("  --- Diagnostics: Evaluating preliminary surrogate model performance on S1 training set ---")
            try:
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

                y_s1_pred_on_train = model.predict_values(x_s1)

                # Ensure y_fit and y_s1_pred_on_train have compatible shapes for comparison
                # y_fit is already in the correct shape for training (usually 1D)
                # y_s1_pred_on_train is also usually 1D

                r2_on_train = r2_score(y_fit, y_s1_pred_on_train)
                mse_on_train = mean_squared_error(y_fit, y_s1_pred_on_train)
                rmse_on_train = np.sqrt(mse_on_train)
                mae_on_train = mean_absolute_error(y_fit, y_s1_pred_on_train)

                print(f"    R2 Score (training set): {r2_on_train:.4f}")
                print(f"    Root Mean Squared Error (RMSE, training set): {rmse_on_train:.4f}")
                print(f"    Mean Absolute Error (MAE, training set): {mae_on_train:.4f}")

                # Compare MAE with the standard deviation of y_fit to see how much better the model is than a simple mean prediction
                y_fit_std = np.std(y_fit)
                print(f"    Standard deviation of target y_fit: {y_fit_std:.4f}")
                if y_fit_std > 1e-9: # Avoid division by zero
                    print(f"    MAE / std(y_fit): {mae_on_train / y_fit_std:.4f} (smaller is better, ideally much less than 1)")
                
                if r2_on_train < 0.1: # A low threshold, can be adjusted depending on the problem
                    print(f"    Warning: The R2 score ({r2_on_train:.4f}) of the preliminary surrogate model on the training set is very low, which may indicate that the model failed to learn the data effectively.")
                elif r2_on_train < 0.5:
                     print(f"    Note: The R2 score ({r2_on_train:.4f}) of the preliminary surrogate model on the training set is moderate. The reliability of SHAP results may be affected.")
                else:
                    print(f"    The R2 score ({r2_on_train:.4f}) of the preliminary surrogate model on the training set is acceptable/good.")

            except Exception as e:
                print(f"    Error during evaluation of the preliminary model: {e}")
            print("  --- End of Diagnostics ---")
            # --- End of evaluation code ---
            
            # Output multi-output information
            if y_s1.shape[1] > 1 and y_fit.ndim > 1 and y_fit.shape[1] > 1:
                print(f"    Model trained for {y_s1.shape[1]} outputs")
            return model
        
        else:
            raise ValueError(f"Unsupported preliminary surrogate model type: {self.model_type}")

    def _perform_shap_analysis(self, model, x_s1: np.ndarray, num_outputs: int):
        """Perform SHAP analysis using the preliminary model"""
        print("  Performing SHAP analysis...")
        
        # Check if model type is suitable for SHAP analysis
        if not hasattr(model, 'feature_importances_') and self.model_type == 'rf':
            print(f"Warning: Model type '{self.model_type}' may not be suitable for SHAP TreeExplainer")

        # Initialize with default values
        shap_values_for_plot = np.zeros_like(x_s1)
        mean_abs_shap = np.ones(self.n_dim)

        try:
            # Perform SHAP analysis
            explainer = shap.TreeExplainer(model)
            raw_shap_values = explainer.shap_values(x_s1)

            # Process SHAP output
            actual_shap_list = []
            
            # Handle list type output
            if isinstance(raw_shap_values, list):
                actual_shap_list = raw_shap_values
                if num_outputs == 1 and len(actual_shap_list) == 1:
                    # Normal for single-output case
                    pass
                elif len(actual_shap_list) != num_outputs:
                    print(f"Warning: SHAP returned a list of length {len(actual_shap_list)} which does not match the expected number of outputs {num_outputs}")
            
            # Handle array type output
            elif isinstance(raw_shap_values, np.ndarray):
                if num_outputs == 1:
                    # Single-output, SHAP directly returns an ndarray (n_samples, n_features)
                    if raw_shap_values.ndim == 2 and raw_shap_values.shape == (x_s1.shape[0], self.n_dim):
                        actual_shap_list = [raw_shap_values]
                    else:
                        raise ValueError(f"Incorrect SHAP array shape for single output: {raw_shap_values.shape}")
                elif num_outputs > 1:
                    # Handle multi-output case
                    actual_shap_list = self._process_multi_output_shap(raw_shap_values, x_s1, num_outputs)
            else:
                raise TypeError(f"SHAP return type {type(raw_shap_values)} is not supported")

            # Validate the list of SHAP values
            if not actual_shap_list:
                raise ValueError("SHAP analysis failed to generate a valid list of values")

            # Calculate feature importance
            all_mean_abs_shap = []
            valid_shap_arrays = []
            
            for i, shap_array in enumerate(actual_shap_list):
                if shap_array is None:
                    print(f"Warning: SHAP values for output {i} are None, skipping")
                    continue
                if not isinstance(shap_array, np.ndarray) or shap_array.shape != (x_s1.shape[0], self.n_dim):
                    print(f"Warning: SHAP values for output {i} have incorrect shape, skipping")
                    continue
                    
                all_mean_abs_shap.append(np.abs(shap_array).mean(axis=0))
                valid_shap_arrays.append(shap_array)

            # Calculate average feature importance
            if not all_mean_abs_shap:
                print("Warning: Failed to calculate SHAP importance. Defaulting to equal importance for all features.")
            else:
                mean_abs_shap = np.mean(np.array(all_mean_abs_shap), axis=0)
                print(f"  SHAP feature importance calculated based on {len(all_mean_abs_shap)} valid outputs")

            # Get SHAP values for plotting
            if valid_shap_arrays:
                shap_values_for_plot = valid_shap_arrays[0]
            else:
                print("Warning: No valid SHAP values available for plotting")

        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            print("Defaulting to equal importance for all features.")

        # Output feature importance
        feature_importance = sorted(zip(self.param_names, mean_abs_shap, range(self.n_dim)),
                                   key=lambda x: x[1], reverse=True)

        print("  SHAP Feature Importance (based on mean absolute SHAP values):")
        for name, imp, _ in feature_importance:
            print(f"    {name}: {imp:.8f}")

        # Get indices of important features
        important_indices = [idx for _, _, idx in feature_importance[:self.n_top_features]]
        print(f"  Indices of top {self.n_top_features} important features: {important_indices}")
        return important_indices, shap_values_for_plot

    def _process_multi_output_shap(self, raw_values, x_s1, num_outputs):
        """Process SHAP values for multi-output cases"""
        result = []
        
        # Shape is (n_outputs, n_samples, n_features)
        if raw_values.ndim == 3 and raw_values.shape[0] == num_outputs and \
           raw_values.shape[1] == x_s1.shape[0] and raw_values.shape[2] == self.n_dim:
            print(f"  SHAP returned shape {raw_values.shape}, parsed as (n_outputs, n_samples, n_features)")
            for i in range(num_outputs):
                result.append(raw_values[i, :, :])
                
        # Shape is (n_samples, n_features, n_outputs)
        elif raw_values.ndim == 3 and raw_values.shape[0] == x_s1.shape[0] and \
             raw_values.shape[1] == self.n_dim and raw_values.shape[2] == num_outputs:
            print(f"  SHAP returned shape {raw_values.shape}, parsed as (n_samples, n_features, n_outputs)")
            for i in range(num_outputs):
                result.append(raw_values[:, :, i])
                
        # Other complex shapes
        elif raw_values.ndim == 2 and raw_values.shape[0] == x_s1.shape[0] and \
             raw_values.shape[1] == self.n_dim * num_outputs:
            print(f"  SHAP returned shape {raw_values.shape}, parsed as (n_samples, n_outputs*n_features)")
            print(f"    Warning: SHAP values with this structure are not currently fully supported for automatic parsing")
        else:
            print(f"  Warning: SHAP returned shape {raw_values.shape} could not be parsed as a multi-output format")
            
        return result

    def _determine_stage2_roi_bounds(self, x_s1: np.ndarray, y_s1: np.ndarray, 
                                     important_indices: list, n_s1: int):
        """Determine Stage 2 ROI bounds based on SHAP results and best samples"""
        print("  Determining Stage 2 ROI bounds...")
        
        # Calculate the number of best samples to determine the ROI
        k_best = max(1, int(self.k_best_factor * n_s1))
        num_outputs = y_s1.shape[1]

        # Determine the values to be used for sorting
        y_for_sorting = None

        # For single-output, use the values directly
        if num_outputs == 1:
            y_for_sorting = y_s1.ravel()
            print(f"  ROI determination: Single-output target, using its values directly for sorting")
        # Multi-output case requires normalization
        else:
            print(f"  ROI determination: Detected {num_outputs} outputs, using normalized mean to determine the best samples")
            print(f"    (Assuming all output targets are to be minimized. If not, please preprocess in the objective_func)")
            
            try:
                # Normalize multi-output
                scaler = MinMaxScaler()
                y_normalized = scaler.fit_transform(y_s1)
                y_for_sorting = np.mean(y_normalized, axis=1)
            except ValueError as e:
                print(f"    Normalization failed: {e}. Will average the original values directly")
                y_for_sorting = np.mean(y_s1, axis=1)
        
        # Get the best samples
        sorted_indices = np.argsort(y_for_sorting)  # Assuming the objective is minimization
        best_samples = x_s1[sorted_indices[:k_best]]

        # Initialize ROI bounds
        lb_roi = self.x_limits_global[:, 0].copy()
        ub_roi = self.x_limits_global[:, 1].copy()

        # Determine ROI bounds for each dimension
        for i in range(self.n_dim):
            # Check if it is an important feature
            is_important = i in important_indices
            padding_factor = self.roi_padding_important if is_important else self.roi_padding_non_important
            
            # Get the bounds for the current dimension
            if best_samples.shape[0] == 0:  # If there are no best samples
                print(f"    Warning: No best sample data for dimension {self.param_names[i]}, using global bounds")
                min_val = self.x_limits_global[i, 0]
                max_val = self.x_limits_global[i, 1]
            else:
                min_val = np.min(best_samples[:, i])
                max_val = np.max(best_samples[:, i])

            # Calculate padding amount
            width = max(0, max_val - min_val)
            min_padding = abs(self.x_limits_global[i, 1] - self.x_limits_global[i, 0]) * \
                         (0.025 if is_important else 0.05)  # Minimum padding amount
                         
            # Determine the actual padding amount
            if width > 1e-9:  # If there is variation in the dimension
                padding = max(width * padding_factor, min_padding)
            else:  # If there is almost no variation in the dimension
                padding = min_padding
            
            # Set ROI bounds
            lb_roi[i] = np.maximum(self.x_limits_global[i, 0], min_val - padding)
            ub_roi[i] = np.minimum(self.x_limits_global[i, 1], max_val + padding)

            # Handle boundary issues (collapsed or inverted bounds)
            if lb_roi[i] >= ub_roi[i]:
                # Use center point and default width
                center = (min_val + max_val) / 2
                half_width = abs(self.x_limits_global[i, 1] - self.x_limits_global[i, 0]) * \
                             (0.01 if is_important else 0.025)
                
                lb_roi[i] = np.maximum(self.x_limits_global[i, 0], center - half_width)
                ub_roi[i] = np.minimum(self.x_limits_global[i, 1], center + half_width)
                
                # Finally, fall back to global bounds
                if lb_roi[i] >= ub_roi[i]:
                    lb_roi[i] = self.x_limits_global[i, 0]
                    ub_roi[i] = self.x_limits_global[i, 1]
        
        # Construct ROI bounds array
        xlimits_s2 = np.vstack((lb_roi, ub_roi)).T
        
        # Example of printing ROI bounds
        # if self.n_dim > 0 and self.param_names:
        #     print(f"  Stage 2 ROI '{self.param_names[0]}' bounds: [{xlimits_s2[0, 0]:.3f}, {xlimits_s2[0, 1]:.3f}] "
        #           f"  Stage 2 ROI '{self.param_names[1]}' bounds: [{xlimits_s2[1, 0]:.3f}, {xlimits_s2[1, 1]:.3f}] "
        #           f"  Stage 2 ROI '{self.param_names[2]}' bounds: [{xlimits_s2[2, 0]:.3f}, {xlimits_s2[2, 1]:.3f}] "
        #           f"  Stage 2 ROI '{self.param_names[3]}' bounds: [{xlimits_s2[3, 0]:.3f}, {xlimits_s2[3, 1]:.3f}] "
        #           )
        if self.n_dim > 0 and self.param_names:
            for i in range(min(self.n_dim, len(self.param_names), xlimits_s2.shape[0])):
                print(f"  Stage 2 ROI '{self.param_names[i]}' bounds: [{xlimits_s2[i, 0]:.8f}, {xlimits_s2[i, 1]:.8f}]")

        return xlimits_s2

    def _perform_stage2_sampling_and_eval(self, xlimits_s2: np.ndarray, n_s2: int):
        """Perform Stage 2 LHS sampling within the ROI and evaluate"""
        print(f"  Stage 2: LHS sampling within ROI ({n_s2} points)...")
        
        # Check and correct ROI bounds
        for i in range(self.n_dim):
            if xlimits_s2[i, 0] > xlimits_s2[i, 1]:
                print(f"    Warning: Invalid ROI bounds for dimension {self.param_names[i]}, using global bounds")
                xlimits_s2[i, :] = self.x_limits_global[i, :]
        
        # Perform sampling
        lhs_s2 = LHS(xlimits=xlimits_s2, criterion='m', random_state=self.s2_seed)
        x_s2 = lhs_s2(n_s2)
        y_s2 = self._evaluate_objective_batch(x_s2)
        
        print(f"  Stage 2 sampling complete. Function calls: {x_s2.shape[0]}")
        return x_s2, y_s2

    def generate_samples(self):
        """Execute the sampling process and return training samples
        
        Returns:
            x_train: Training set input features (n_samples, n_features)
            y_train: Training set output values (n_samples, n_outputs)
            total_calls: Total number of calls to the objective function
            shap_details: Information for SHAP plotting (returned only by the two-stage strategy)
        """
        # Parameter validation
        if self.n_samples_s1_factor <= 0:
            raise ValueError("n_samples_s1_factor must be a positive number")

        # Single-stage sampling
        if self.n_samples_s2_factor <= 0:
            return self._perform_single_stage_lhs()

        # Two-stage SHAP-guided sampling
        print(f"\n--- SGTS_LHS: Executing SHAP-guided two-stage strategy ---")
        n_s1 = self.n_samples_s1_factor * self.n_dim  # Number of samples for Stage 1
        n_s2 = self.n_samples_s2_factor * self.n_dim  # Number of samples for Stage 2
        
        # Stage 1: Sampling and evaluation
        x_s1, y_s1 = self._perform_stage1_sampling_and_eval(n_s1)
        num_outputs = y_s1.shape[1]
        
        # Train preliminary surrogate model
        prelim_model = self._train_preliminary_surrogate(x_s1, y_s1)
        
        # SHAP analysis
        important_indices, shap_values = self._perform_shap_analysis(prelim_model, x_s1, num_outputs)
        
        # Determine ROI bounds
        xlimits_s2 = self._determine_stage2_roi_bounds(x_s1, y_s1, important_indices, n_s1)
        
        # Initialize training data
        x_train, y_train = x_s1, y_s1
        
        # Stage 2: Sampling within ROI
        if n_s2 > 0:
            x_s2, y_s2 = self._perform_stage2_sampling_and_eval(xlimits_s2, n_s2)
            x_train = np.vstack((x_s1, x_s2))  # Combine samples
            y_train = np.vstack((y_s1, y_s2))  # Combine outputs
        else:
            print("  Number of samples for Stage 2 is 0, skipping Stage 2 sampling")
        
        # Consolidate and return results
        total_calls = x_train.shape[0]
        print(f"  Total function calls for SHAP-guided two-stage strategy: {total_calls}")

        # Return SHAP details for plotting
        shap_details = {
            "shap_values": shap_values, 
            "x_samples": x_s1, 
            "feature_names": self.param_names
        }
        return x_train, y_train, total_calls, shap_details