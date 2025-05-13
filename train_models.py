# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import argparse
import os
import joblib # For saving models

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
    3. Drops rows with any NaNs in the selected features (often from lags).
    """
    # 1. Drop rows where the target is NaN (these are typically at the end of each battery's sequence)
    df_cleaned = df.dropna(subset=[target_col_name]).copy()
    if df_cleaned.empty:
        print(f"Error: No data remaining after dropping NaNs in target column '{target_col_name}'. Check forecast horizon and data.")
        return None, None, None, None

    y = df_cleaned[target_col_name]

    # 2. Define feature columns
    # Exclude identifiers, the direct SOH measure (as it's what we are trying to predict indirectly),
    # and the target itself.
    # Also exclude 'q_initial_Ah' if SOH is already normalized by it, or if it's constant.
    # 'battery_id' is kept for now for grouped splitting, will be dropped before training.
    excluded_cols = [
        target_col_name,
        'SOH_cycle_capacity_%', # This is the current SOH, not the target future SOH
        'start_time', # If present from older CSV versions
        # 'q_initial_Ah', # Could be useful if training on multiple batteries and q_initial varies
                         # but SOH target is already normalized. Let's exclude for now.
    ]
    
    # Identify potential feature columns
    potential_feature_cols = [col for col in df_cleaned.columns if col not in excluded_cols]
    
    # Select only numeric feature columns for now (or handle categoricals like is_reference_cycle later if needed)
    # 'battery_id' is not numeric but needed for splitting, so handle it specially
    numeric_features = df_cleaned[potential_feature_cols].select_dtypes(include=np.number).columns.tolist()
    
    # Ensure 'battery_id' is present for splitting if it exists, then select X
    if 'battery_id' in df_cleaned.columns:
        feature_cols_for_X = [col for col in numeric_features if col != 'battery_id']
        X = df_cleaned[['battery_id'] + feature_cols_for_X].copy() # Keep battery_id for splitting
    else:
        feature_cols_for_X = numeric_features
        X = df_cleaned[feature_cols_for_X].copy()

    print(f"Target variable: {target_col_name}")
    print(f"Number of potential features (before final NaN drop): {len(feature_cols_for_X)}")

    # 3. Drop rows with any NaNs in the selected feature set X
    # This handles NaNs from lag features at the beginning of sequences.
    # Align y with X after this drop
    X_original_index = X.index
    X.dropna(inplace=True)
    y = y.loc[X.index] # Align y with the rows remaining in X

    if X.empty:
        print("Error: No data remaining after dropping NaNs in features. Check lag feature creation.")
        return None, None, None, None

    print(f"Shape of X after NaN drop in features: {X.shape}")
    print(f"Shape of y after alignment: {y.shape}")
    
    # Store battery_ids for grouped splitting before dropping the column from X
    battery_ids_for_split = X['battery_id'] if 'battery_id' in X.columns else None
    if 'battery_id' in X.columns:
        X_final_features = X.drop(columns=['battery_id'])
    else:
        X_final_features = X
        
    return X_final_features, y, battery_ids_for_split, X_final_features.columns.tolist()

def split_data(X, y, battery_ids, test_size=0.2, validation_size=0.15, random_state=42):
    """
    Splits data into train, validation, and test sets.
    If battery_ids are provided, performs a grouped split.
    Otherwise, performs a chronological-like split (if data is sorted, though less robust without explicit time).
    For simplicity here, if no battery_ids, it will be a random split which is NOT ideal for time series.
    Proper chronological split for single battery needs sorting by cycle_number before this function.
    """
    if X.empty or y.empty:
        print("Error: X or y is empty before splitting.")
        return None
        
    if battery_ids is not None and len(battery_ids.unique()) > 1:
        print("Performing grouped split based on battery_id.")
        # Split to get train and temp (validation + test)
        gss_train_temp = GroupShuffleSplit(n_splits=1, test_size=test_size + validation_size, random_state=random_state)
        train_idx, temp_idx = next(gss_train_temp.split(X, y, groups=battery_ids))
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]
        groups_temp = battery_ids.iloc[temp_idx]

        # Split temp into validation and test
        # Adjust validation_size relative to the size of X_temp
        relative_validation_size = validation_size / (test_size + validation_size)
        gss_val_test = GroupShuffleSplit(n_splits=1, test_size=1-relative_validation_size, random_state=random_state) # test_size is effectively 1-val_size
        if len(groups_temp.unique()) > 1 : # Ensure enough groups for another split
            val_idx, test_idx = next(gss_val_test.split(X_temp, y_temp, groups=groups_temp))
            X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
            X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]
        else: # Not enough groups in temp for further grouped split, do a random one or assign all to test/val
            print("Warning: Not enough unique battery_ids in temp set for further grouped split. Splitting randomly or assigning.")
            if len(X_temp) * relative_validation_size < 1: # if val set would be empty
                X_val, y_val = pd.DataFrame(columns=X.columns), pd.Series(dtype=y.dtype) # empty val
                X_test, y_test = X_temp, y_temp
            else:
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-relative_validation_size, random_state=random_state)


    elif 'cycle_number' in X.columns: # Assuming single battery, or pre-sorted data
        print("Performing chronological-like split (assuming data is sorted by cycle_number).")
        # This simplified split assumes data is already sorted by cycle for a single battery.
        # For multiple batteries without group split, this is not ideal.
        # Create a combined test + validation set first
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + validation_size), shuffle=False
        )
        # Split the combined set into validation and test
        # Adjust validation_size relative to the size of X_temp
        if not X_temp.empty:
            relative_validation_size = validation_size / (test_size + validation_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=1-relative_validation_size, shuffle=False # (1 - relative_validation_size) is the new test size
            )
        else: # X_temp is empty
            X_val, y_val = pd.DataFrame(columns=X.columns), pd.Series(dtype=y.dtype)
            X_test, y_test = pd.DataFrame(columns=X.columns), pd.Series(dtype=y.dtype)
    else:
        print("Warning: No battery_id for grouping and no cycle_number for chronological split. Performing random split (NOT ideal for time series).")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + validation_size, random_state=random_state, shuffle=True)
        relative_validation_size = validation_size / (test_size + validation_size)
        if not X_temp.empty:
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-relative_validation_size, random_state=random_state, shuffle=True)
        else:
             X_val, y_val = pd.DataFrame(columns=X.columns), pd.Series(dtype=y.dtype)
             X_test, y_test = pd.DataFrame(columns=X.columns), pd.Series(dtype=y.dtype)


    print(f"Train set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0] if not X_val.empty else 0}")
    print(f"Test set size: {X_test.shape[0] if not X_test.empty else 0}")
    
    if X_train.empty or (X_val.empty and X_test.empty):
        print("Error: Training set or both validation/test sets are empty after split.")
        return None
        
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test):
    """Scales features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Handle empty validation or test sets before transforming
    X_val_scaled = scaler.transform(X_val) if not X_val.empty else X_val 
    X_test_scaled = scaler.transform(X_test) if not X_test.empty else X_test
    
    # Convert back to DataFrame to keep column names (optional, but good for inspection)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    if not X_val.empty:
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    if not X_test.empty:
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, models_to_train=None):
    """Trains specified models and evaluates them."""
    if models_to_train is None:
        models_to_train = ['rf', 'gb', 'lr', 'xgb'] # Default models

    results = {}
    trained_models_dict = {}

    for model_name in models_to_train:
        print(f"\nTraining {model_name.upper()}...")
        if model_name == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20, min_samples_split=5, min_samples_leaf=2)
        elif model_name == 'gb':
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, min_samples_split=5, min_samples_leaf=2)
        elif model_name == 'lr':
            model = LinearRegression()
        elif model_name == 'xgb':
            model = xgb.XGBRegressor(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0  # L2 regularization
            )
        else:
            print(f"Warning: Unknown model '{model_name}'. Skipping.")
            continue

        model.fit(X_train, y_train)
        trained_models_dict[model_name] = model

        print("Evaluating on Validation Set...")
        if not X_val.empty:
            y_pred_val = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            val_mae = mean_absolute_error(y_val, y_pred_val)
            val_r2 = r2_score(y_val, y_pred_val)
            print(f"  Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")
        else:
            print("  Validation set is empty. Skipping validation metrics.")
            val_rmse, val_mae, val_r2 = np.nan, np.nan, np.nan


        print("Evaluating on Test Set...")
        if not X_test.empty:
            y_pred_test = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            print(f"  Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")
        else:
            print("  Test set is empty. Skipping test metrics.")
            test_rmse, test_mae, test_r2 = np.nan, np.nan, np.nan


        results[model_name] = {
            'val_rmse': val_rmse, 'val_mae': val_mae, 'val_r2': val_r2,
            'test_rmse': test_rmse, 'test_mae': test_mae, 'test_r2': test_r2
        }

        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_') and feature_names:
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(15)
            print(f"\n  Top 15 Feature Importances for {model_name.upper()}:")
            print(feature_importance_df)

    return results, trained_models_dict

def main(args):
    df = load_data(args.input_file)
    if df is None:
        return

    target_col = f"SOH_target_h{args.forecast_horizon}"
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in the DataFrame.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    X_processed, y_processed, battery_ids_for_split, feature_names_list = preprocess_data(df, target_col)
    if X_processed is None or y_processed is None:
        print("Failed to preprocess data.")
        return

    # Ensure cycle_number is in X_processed if it exists, for chronological split logic
    # The feature selection in preprocess_data should already handle this.
    
    split_results = split_data(X_processed, y_processed, battery_ids_for_split, 
                               test_size=args.test_size, validation_size=args.val_size)
    if split_results is None:
        print("Failed to split data.")
        return
    X_train, X_val, X_test, y_train, y_val, y_test = split_results

    if X_train.empty:
        print("Training data is empty. Exiting.")
        return
        
    # It's possible for X_val or X_test to be empty if the dataset is very small
    # or if group splitting results in empty sets for specific groups.
    # The scaling and evaluation functions should handle empty DataFrames.

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Save the scaler
    scaler_filename = f"scaler_h{args.forecast_horizon}.joblib"
    joblib.dump(scaler, scaler_filename)
    print(f"\nScaler saved to {scaler_filename}")


    model_results, trained_models = train_and_evaluate_models(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test,
        feature_names=feature_names_list, # Pass the actual feature names
        models_to_train=args.models
    )

    print("\n--- Final Results Summary ---")
    for model_name, metrics in model_results.items():
        print(f"\nModel: {model_name.upper()}")
        print(f"  Validation: RMSE={metrics['val_rmse']:.4f}, MAE={metrics['val_mae']:.4f}, R2={metrics['val_r2']:.4f}")
        print(f"  Test:       RMSE={metrics['test_rmse']:.4f}, MAE={metrics['test_mae']:.4f}, R2={metrics['test_r2']:.4f}")

    # Save trained models
    if args.save_models:
        output_model_dir = "trained_models"
        os.makedirs(output_model_dir, exist_ok=True)
        for model_name, model_instance in trained_models.items():
            model_filename = os.path.join(output_model_dir, f"{model_name}_h{args.forecast_horizon}.joblib")
            joblib.dump(model_instance, model_filename)
            print(f"Trained model {model_name.upper()} saved to {model_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SOH forecasting models.")
    parser.add_argument("input_file", type=str, help="Path to the processed CSV data file.")
    parser.add_argument("--forecast_horizon", type=int, default=5, help="Forecast horizon for SOH target (e.g., 5 for SOH_target_h5).")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data for the test set.")
    parser.add_argument("--val_size", type=float, default=0.15, help="Proportion of data for the validation set (taken from non-test data).")
    parser.add_argument("--models", nargs='+', default=['rf', 'gb', 'lr', 'xgb'], help="List of models to train (e.g., rf gb lr).")
    parser.add_argument("--save_models", action='store_true', help="Save trained models to disk.")


    args = parser.parse_args()
    main(args)