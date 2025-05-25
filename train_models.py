# train_models.py

import pandas as pd
import numpy as np
# GroupShuffleSplit might not be directly used if splitting is fully manual in notebook
# from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import os
import joblib

def load_data(filepath):
    """Loads data from the specified CSV file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df, target_col_name):
    """
    Prepares data for training:
    1. Drops rows with NaN in the target column.
    2. Separates features (X) and target (y).
    3. Selects features, converts types if necessary.
    """
    if df is None or df.empty:
        print("Error: Input DataFrame to preprocess_data is None or empty.")
        return None, None, None, None
        
    df_cleaned = df.dropna(subset=[target_col_name]).copy()
    if df_cleaned.empty:
        print(f"Error: No data remaining after dropping NaNs in target column '{target_col_name}'.")
        return None, None, None, None

    y = df_cleaned[target_col_name]

    # Define columns to always exclude from features
    # If target_col_name is 'SOH_cycle_capacity_%', it will be correctly excluded.
    # If we are forecasting (target is SOH_target_hX), current SOH is also excluded.
    base_excluded_cols = [
        'start_time',
        'cycle_number',
    ]
    # Add the target column itself and SOH_cycle_capacity_% (if it's not the target)
    # to the list of columns to remove from features.
    excluded_cols_from_features = list(set(base_excluded_cols + [target_col_name, 'SOH_cycle_capacity_%']))


    potential_feature_cols_candidates = [col for col in df_cleaned.columns if col not in excluded_cols_from_features]
    
    temp_X_df = df_cleaned.copy() # Work on a copy for type conversions

    if 'is_reference_cycle' in temp_X_df.columns:
        temp_X_df['is_reference_cycle'] = temp_X_df['is_reference_cycle'].astype(int)

    feature_cols_for_X = []
    for col in potential_feature_cols_candidates:
        if col == 'battery_id': # battery_id is handled for grouping, not a direct model feature
            continue
        if temp_X_df[col].dtype == 'object':
            # Attempt to convert regime or other known object columns if they slip through,
            # otherwise warn.
            if col == 'regime': # Example: if regime was accidentally not dropped earlier
                 print(f"Warning: 'regime' column found in feature candidates. It should be handled separately. Skipping.")
                 continue
            print(f"Warning: Skipping object type column '{col}' during feature selection.")
            continue
        feature_cols_for_X.append(col)
    
    feature_cols_for_X = sorted(list(set(feature_cols_for_X)))

    # Create X. 'battery_id' is not included in X passed to models.
    # The calling notebook function (run_experiment) handles splitting by battery_id.
    X = temp_X_df[feature_cols_for_X].copy()
    # battery_ids_for_split is not strictly needed by the notebook's run_experiment
    # as it pre-filters, but retaining for potential other uses or more direct script runs.
    battery_ids_for_split = temp_X_df['battery_id'] if 'battery_id' in temp_X_df.columns else None


    print(f"Target variable: {target_col_name}")
    print(f"Number of features selected: {len(feature_cols_for_X)}")
    # print(f"Selected features for X: {feature_cols_for_X}")

    if X.empty:
        print("Error: X is empty after feature selection.")
        return None, None, None, None
    if y.empty:
        print("Error: y is empty.")
        return None, None, None, None

    print(f"Shape of X (features for model): {X.shape}")
    print(f"Shape of y: {y.shape}")
        
    return X, y, battery_ids_for_split, feature_cols_for_X # Return original column names

def scale_features(X_train, X_val, X_test):
    """
    Scales features using StandardScaler.
    Assumes X_train, X_val, X_test are DataFrames and already imputed.
    """
    scaler = StandardScaler()
    
    X_train_cols, X_train_idx = X_train.columns, X_train.index
    
    X_train_scaled_np = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=X_train_cols, index=X_train_idx)
    
    X_val_scaled = pd.DataFrame(columns=X_train.columns, index=X_val.index if X_val is not None else None, dtype=float)
    if X_val is not None and not X_val.empty:
        X_val_cols, X_val_idx = X_val.columns, X_val.index
        X_val_scaled_np = scaler.transform(X_val)
        X_val_scaled = pd.DataFrame(X_val_scaled_np, columns=X_val_cols, index=X_val_idx)
    
    X_test_scaled = pd.DataFrame(columns=X_train.columns, index=X_test.index if X_test is not None else None, dtype=float)
    if X_test is not None and not X_test.empty:
        X_test_cols, X_test_idx = X_test.columns, X_test.index
        X_test_scaled_np = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=X_test_cols, index=X_test_idx)
        
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def train_and_evaluate_models(
    X_train_imputed, y_train, 
    X_val_imputed, y_val, 
    X_test_imputed, y_test,
    X_train_scaled, 
    X_val_scaled, 
    X_test_scaled,
    feature_names,  # Pass the actual feature names used for training
    models_to_train=None
):
    """
    Trains specified models with default parameters and evaluates them.
    RF and XGB use imputed (unscaled) data.
    LR and HistGBM use scaled data.
    """
    if models_to_train is None:
        # Defaulting to all, including 'lr' as requested.
        models_to_train = ['rf', 'gb', 'lr', 'xgb'] 

    results = {}
    trained_models_dict = {}

    for model_name in models_to_train:
        print(f"\nTraining {model_name.upper()} with default parameters...")
        
        current_X_train = X_train_imputed
        current_X_val = X_val_imputed
        current_X_test = X_test_imputed

        if model_name in ['lr', 'gb']: # HistGradientBoostingRegressor for 'gb'
            current_X_train = X_train_scaled
            current_X_val = X_val_scaled
            current_X_test = X_test_scaled
            print(f"  Using SCALED data for {model_name.upper()}")
        else:
            print(f"  Using IMPUTED (unscaled) data for {model_name.upper()}")

        # Initialize models with default parameters
        if model_name == 'rf':
            model = RandomForestRegressor(random_state=42, n_jobs=-1) # Common to set random_state and n_jobs
        elif model_name == 'gb': # HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(random_state=42) # Early stopping is 'auto' by default
        elif model_name == 'lr':
            model = LinearRegression()
        elif model_name == 'xgb':
            model = xgb.XGBRegressor(random_state=42, n_jobs=-1) # Common to set random_state and n_jobs
        else:
            print(f"Warning: Unknown model '{model_name}'. Skipping.")
            continue

        # Fit model
        # HistGBM handles its own validation split for early stopping by default.
        # For XGBoost, to be strictly "default parameters", we don't pass eval_set to fit,
        # as early_stopping_rounds is not a default constructor param and must be specified for fit.
        # If an explicit validation set IS provided to this function, and the model
        # supports it for early stopping (like HistGBM by default or XGB with params), it's used.
        # For XGBoost, without explicit early_stopping_rounds in fit(), eval_set is for monitoring only.
        if model_name == 'xgb' and not current_X_val.empty and not y_val.empty:
            print("  Fitting XGBoost. Validation set provided but early_stopping_rounds not specified for strict default fit.")
            # For strict default fit without explicit early stopping rounds:
            model.fit(current_X_train, y_train)
            # If you wanted to use eval_set for potential early stopping, you'd need early_stopping_rounds:
            # model.fit(current_X_train, y_train, eval_set=[(current_X_val, y_val)], early_stopping_rounds=10, verbose=False)
        elif model_name == 'gb': # HistGradientBoostingRegressor
             print("  Fitting HistGradientBoostingRegressor (default early stopping if applicable).")
             model.fit(current_X_train, y_train)
        else:
            model.fit(current_X_train, y_train)
            
        trained_models_dict[model_name] = model

        # Evaluation on Validation Set
        val_rmse, val_mae, val_r2 = np.nan, np.nan, np.nan
        if not current_X_val.empty and not y_val.empty:
            try:
                y_pred_val = model.predict(current_X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                val_mae = mean_absolute_error(y_val, y_pred_val)
                val_r2 = r2_score(y_val, y_pred_val)
                print(f"  Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")
            except Exception as e:
                print(f"  Error during validation evaluation for {model_name}: {e}")
        else:
            print("  Validation set is empty or y_val is empty. Skipping validation metrics.")

        # Evaluation on Test Set
        test_rmse, test_mae, test_r2 = np.nan, np.nan, np.nan
        if not current_X_test.empty and not y_test.empty:
            try:
                y_pred_test = model.predict(current_X_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                test_mae = mean_absolute_error(y_test, y_pred_test)
                test_r2 = r2_score(y_test, y_pred_test)
                print(f"  Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")
            except Exception as e:
                print(f"  Error during test evaluation for {model_name}: {e}")
        else:
            print("  Test set is empty or y_test is empty. Skipping test metrics.")

        results[model_name] = {
            'val_rmse': val_rmse, 'val_mae': val_mae, 'val_r2': val_r2,
            'test_rmse': test_rmse, 'test_mae': test_mae, 'test_r2': test_r2
        }

        # Feature Importances
        # Use the passed 'feature_names' which should correspond to the columns of X_train_imputed/scaled
        if feature_names:
            importances_values = None
            if hasattr(model, 'feature_importances_'):
                importances_values = model.feature_importances_
            elif model_name == 'lr' and hasattr(model, 'coef_'):
                importances_values = np.abs(model.coef_)
                if len(importances_values.shape) > 1: # If multi-target (not expected here)
                    importances_values = np.mean(importances_values, axis=0)
            
            if importances_values is not None:
                if len(feature_names) == len(importances_values):
                    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances_values})
                    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(15)
                    print(f"\n  Top 15 Feature Importances for {model_name.upper()}:")
                    print(feature_importance_df)
                else:
                    print(f"Warning: Mismatch between number of feature names ({len(feature_names)}) and importances ({len(importances_values)}) for model {model_name}.")
            elif model_name not in ['lr']: # LR handled by coef_, others should have feature_importances_
                 print(f"Warning: Model {model_name} does not have 'feature_importances_' or 'coef_' attribute for feature ranking.")

    return results, trained_models_dict

# Removed main() function and argparse, as the script is now primarily a library for the notebook.
# The notebook's 'run_experiment' function will call these functions directly.