# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor # Changed from GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer # For handling NaNs
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
    3. Selects features, converts types if necessary. NaNs from lags are kept for imputation later.
    """
    df_cleaned = df.dropna(subset=[target_col_name]).copy()
    if df_cleaned.empty:
        print(f"Error: No data remaining after dropping NaNs in target column '{target_col_name}'. Check forecast horizon and data.")
        return None, None, None, None

    y = df_cleaned[target_col_name]

    excluded_cols = [
        target_col_name,
        'SOH_cycle_capacity_%',
        'start_time',
        'cycle_number', # Exclude cycle_number as a direct feature
        # 'q_initial_Ah', # Already excluded in original
    ]
    
    potential_feature_cols_candidates = [col for col in df_cleaned.columns if col not in excluded_cols]
    
    # Use a temporary copy of df_cleaned for type conversion before selecting X
    temp_X_df = df_cleaned.copy()

    # Convert boolean 'is_reference_cycle' to int if present
    if 'is_reference_cycle' in temp_X_df.columns:
        temp_X_df['is_reference_cycle'] = temp_X_df['is_reference_cycle'].astype(int)

    # Select final feature columns for X
    # We will include all non-excluded columns that are not 'object' type, plus 'is_reference_cycle' (now int) and 'mission_type' (should be int)
    feature_cols_for_X = []
    for col in potential_feature_cols_candidates:
        if col == 'battery_id': # battery_id is handled separately, not a direct model feature
            continue
        if temp_X_df[col].dtype == 'object': # Skip any remaining object columns
            print(f"Warning: Skipping object type column '{col}' during feature selection in preprocess_data.")
            continue
        feature_cols_for_X.append(col)
    
    feature_cols_for_X = sorted(list(set(feature_cols_for_X))) # Ensure uniqueness and order

    # Create X, keeping battery_id for splitting if it exists
    if 'battery_id' in temp_X_df.columns:
        X = temp_X_df[['battery_id'] + feature_cols_for_X].copy()
    else:
        X = temp_X_df[feature_cols_for_X].copy()

    print(f"Target variable: {target_col_name}")
    print(f"Number of features selected (before final battery_id drop from X): {len(feature_cols_for_X)}")
    # print(f"Selected features for X (excluding battery_id for model training): {feature_cols_for_X}")


    if X.empty:
        print("Error: X is empty after feature selection. Check input data and exclusions.")
        return None, None, None, None
    if y.empty: # Should be caught by df_cleaned.empty check
        print("Error: y is empty.")
        return None, None, None, None

    print(f"Shape of X (may contain NaNs from lags, battery_id included for now): {X.shape}")
    print(f"Shape of y: {y.shape}")
    
    battery_ids_for_split = X['battery_id'] if 'battery_id' in X.columns else None
    
    # X_final_features will be used for training after battery_id is dropped
    if 'battery_id' in X.columns:
        X_final_features = X.drop(columns=['battery_id'])
    else:
        X_final_features = X
        
    return X_final_features, y, battery_ids_for_split, X_final_features.columns.tolist()

def split_data(X, y, battery_ids, test_size=0.2, validation_size=0.15, random_state=42, strategy='chronological'):
    """
    Splits data into train, validation, and test sets.
    'chronological': Assumes X, y are pre-sorted (e.g., by battery_id, then cycle_number/time).
                     Splits entire dataset chronologically.
    'group_shuffle_batteries': Uses GroupShuffleSplit to put entire batteries into different sets.
                               Tests generalization to unseen batteries.
    """
    if X.empty or y.empty:
        print("Error: X or y is empty before splitting.")
        return None, None, None, None, None, None # Ensure 6 return values for unpacking

    X_train, X_val, X_test = pd.DataFrame(columns=X.columns), pd.DataFrame(columns=X.columns), pd.DataFrame(columns=X.columns)
    y_train, y_val, y_test = pd.Series(dtype=y.dtype, name=y.name), pd.Series(dtype=y.dtype, name=y.name), pd.Series(dtype=y.dtype, name=y.name)

    if strategy == 'chronological':
        print("Performing chronological split (assuming data is pre-sorted by battery, then time/cycle).")
        # This split ensures that for any given battery, its test data comes after its train/validation data.
        
        # First, split off a combined (validation + test) set chronologically from the end.
        # Ensure there's enough data for the split
        if len(X) * (test_size + validation_size) < 1 and len(X) > 0 : # Not enough for temp set but some data exists
             print("Warning: Not enough data for a full chronological val+test split. Assigning all to train or adjusting sizes.")
             if len(X) * test_size < 1: # Not enough for test either
                X_train, y_train = X, y
                # X_val, y_val, X_test, y_test remain empty
             else: # Enough for test, but not val
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
                # X_val, y_val remain empty
        elif len(X) == 0:
            print("Error: X is empty, cannot split.")
            # All remain empty
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=(test_size + validation_size), shuffle=False
            )
            # Then, split the temp set into validation and test, also chronologically.
            if not X_temp.empty:
                # Calculate the proportion of the temp set that should be test data
                # to achieve the original overall test_size.
                relative_test_size_in_temp = test_size / (test_size + validation_size)
                if len(X_temp) * relative_test_size_in_temp < 1 and len(X_temp) > 0: # Not enough in temp for test split
                    print("Warning: Not enough data in temp set for test split. Assigning all temp to validation.")
                    X_val, y_val = X_temp, y_temp
                    # X_test, y_test remain empty
                elif len(X_temp) == 0: # Should not happen if X_temp was not empty check worked
                    pass # X_val, X_test remain empty
                else:
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp, y_temp, test_size=relative_test_size_in_temp, shuffle=False
                    )
            # If X_temp was empty, X_val, X_test remain empty, which is handled.

    elif strategy == 'group_shuffle_batteries':
        if battery_ids is None or len(battery_ids.unique()) <= 1:
            print("Warning: GroupShuffleSplit strategy selected, but no/few unique battery_ids. Falling back to random split.")
            # Fallback to random split if groups are insufficient (this part mirrors original 'else' logic)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + validation_size, random_state=random_state, shuffle=True)
            if not X_temp.empty:
                relative_validation_size = validation_size / (test_size + validation_size) if (test_size + validation_size) > 0 else 0
                if len(X_temp) * (1-relative_validation_size) < 1 and len(X_temp) > 0 : # Not enough for test after taking val
                     X_val, y_val = X_temp, y_temp # Assign all temp to val
                elif len(X_temp) == 0:
                    pass
                else:
                    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-relative_validation_size, random_state=random_state, shuffle=True)
            # else X_val, X_test remain empty
        else:
            print("Performing grouped shuffle split based on battery_id (for unseen battery generalization).")
            gss_train_temp = GroupShuffleSplit(n_splits=1, test_size=test_size + validation_size, random_state=random_state)
            train_idx, temp_idx = next(gss_train_temp.split(X, y, groups=battery_ids))
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]
            groups_temp = battery_ids.iloc[temp_idx]

            if not X_temp.empty:
                relative_validation_size = validation_size / (test_size + validation_size) if (test_size + validation_size) > 0 else 0
                
                if len(groups_temp.unique()) > 1 :
                    # test_size for GSS is the size of the second group (test here)
                    # So, if we want validation_size to be relative_validation_size of temp, then test for GSS is 1 - relative_validation_size
                    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=1-relative_validation_size, random_state=random_state)
                    val_idx, test_idx = next(gss_val_test.split(X_temp, y_temp, groups=groups_temp))
                    X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
                    X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]
                else:
                    print("Warning: Not enough unique battery_ids in temp set for further grouped split. Splitting randomly within temp.")
                    if len(X_temp) * (1-relative_validation_size) < 1 and len(X_temp) > 0:
                        X_val, y_val = X_temp, y_temp
                    elif len(X_temp) == 0:
                        pass
                    else:
                        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-relative_validation_size, random_state=random_state, shuffle=True)
            # else X_val, X_test remain empty
    else:
        print(f"Unknown split strategy: {strategy}. Falling back to random split.")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + validation_size, random_state=random_state, shuffle=True)
        if not X_temp.empty:
            relative_validation_size = validation_size / (test_size + validation_size) if (test_size + validation_size) > 0 else 0
            if len(X_temp) * (1-relative_validation_size) < 1 and len(X_temp) > 0 :
                 X_val, y_val = X_temp, y_temp
            elif len(X_temp) == 0:
                pass
            else:
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-relative_validation_size, random_state=random_state, shuffle=True)
        # else X_val, X_test remain empty


    print(f"Train set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0] if not X_val.empty else 0}")
    print(f"Test set size: {X_test.shape[0] if not X_test.empty else 0}")
    
    if X_train.empty and (X_val.empty and X_test.empty) and not X.empty : # If X was not empty, but splits are, it's an issue.
        print("Error: Training set or both validation/test sets are empty after split, but original data was not.")
        # Return empty dataframes with correct structure if X was originally empty
        return (pd.DataFrame(columns=X.columns), pd.DataFrame(columns=X.columns), pd.DataFrame(columns=X.columns),
                pd.Series(dtype=y.dtype, name=y.name), pd.Series(dtype=y.dtype, name=y.name), pd.Series(dtype=y.dtype, name=y.name))

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Scales features using StandardScaler. Assumes X_train, X_val, X_test are already imputed."""
    scaler = StandardScaler()
    
    # Store columns and index to restore DataFrame structure
    X_train_cols, X_train_idx = X_train.columns, X_train.index
    X_val_cols, X_val_idx = X_val.columns, X_val.index
    X_test_cols, X_test_idx = X_test.columns, X_test.index

    X_train_scaled_np = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=X_train_cols, index=X_train_idx)
    
    X_val_scaled = pd.DataFrame(columns=X_val_cols, index=X_val_idx, dtype=float)
    if not X_val.empty:
        X_val_scaled_np = scaler.transform(X_val)
        X_val_scaled = pd.DataFrame(X_val_scaled_np, columns=X_val_cols, index=X_val_idx)
    
    X_test_scaled = pd.DataFrame(columns=X_test_cols, index=X_test_idx, dtype=float)
    if not X_test.empty:
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
    feature_names, 
    models_to_train=None
):
    """Trains specified models and evaluates them.
       RF and XGB use imputed (unscaled) data.
       LR and HistGBM use scaled data.
    """
    if models_to_train is None:
        models_to_train = ['rf', 'gb', 'lr', 'xgb'] # Default models, gb is now HistGBM

    results = {}
    trained_models_dict = {}

    for model_name in models_to_train:
        print(f"\nTraining {model_name.upper()}...")
        
        # Select appropriate dataset (scaled or unscaled)
        current_X_train = X_train_imputed
        current_X_val = X_val_imputed
        current_X_test = X_test_imputed

        if model_name in ['lr', 'gb']: # gb is HistGradientBoostingRegressor
            current_X_train = X_train_scaled
            current_X_val = X_val_scaled
            current_X_test = X_test_scaled
            print(f"  Using SCALED data for {model_name.upper()}")
        else:
            print(f"  Using IMPUTED (unscaled) data for {model_name.upper()}")


        if model_name == 'rf':
            model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1, 
                max_depth=20, 
                min_samples_split=5, 
                min_samples_leaf=2,
                max_samples=0.8 # Added
            )
        elif model_name == 'gb': # Now HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(
                max_iter=1000,          # Number of trees
                learning_rate=0.01,     # Lower learning rate
                max_depth=None,         # Default, often performs well. Or set e.g. 5
                random_state=42,
                early_stopping=True,    # Enable early stopping
                validation_fraction=0.1,# Fraction of training data for internal validation
                n_iter_no_change=50,    # Rounds for early stopping patience
                l2_regularization=0.1   # Example regularization
            )
        elif model_name == 'lr':
            model = LinearRegression()
        elif model_name == 'xgb':
            model = xgb.XGBRegressor(
                n_estimators=1000,        # Increased
                learning_rate=0.01,       # Decreased
                max_depth=5,              # Kept, but can be tuned
                random_state=42,
                n_jobs=-1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                subsample=0.8,            # Added
                colsample_bytree=0.8,     # Added
                early_stopping_rounds=50
            )
        else:
            print(f"Warning: Unknown model '{model_name}'. Skipping.")
            continue

        # Fit model
        if model_name == 'xgb' and not current_X_val.empty:
            print("  Fitting XGBoost with early stopping using explicit validation set.")
            model.fit(current_X_train, y_train, 
                      eval_set=[(current_X_val, y_val)], 
                      verbose=False)
        elif model_name == 'gb' and hasattr(model, 'early_stopping') and model.early_stopping:
             print("  Fitting HistGradientBoostingRegressor with early stopping (uses internal validation split from train data).")
             model.fit(current_X_train, y_train) # HGBR handles its own val split for early stopping
        else:
            model.fit(current_X_train, y_train)
            
        trained_models_dict[model_name] = model

        print("Evaluating on Validation Set...")
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
            print("  Validation set is empty. Skipping validation metrics.")


        print("Evaluating on Test Set...")
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
            print("  Test set is empty. Skipping test metrics.")

        results[model_name] = {
            'val_rmse': val_rmse, 'val_mae': val_mae, 'val_r2': val_r2,
            'test_rmse': test_rmse, 'test_mae': test_mae, 'test_r2': test_r2
        }

        if hasattr(model, 'feature_importances_') and feature_names:
            importances = model.feature_importances_
            # For Linear Regression, use model.coef_
            if model_name == 'lr':
                 # Absolute value of coefficients for importance ranking
                importances = np.abs(model.coef_)
                if len(importances.shape) > 1: # If multi-target, take mean or first
                    importances = np.mean(importances, axis=0)


            if len(feature_names) == len(importances):
                feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(15)
                print(f"\n  Top 15 Feature Importances for {model_name.upper()}:")
                print(feature_importance_df)
            else:
                print(f"Warning: Mismatch between number of feature names ({len(feature_names)}) and importances ({len(importances)}) for model {model_name}.")


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
    
    split_results = split_data(X_processed, y_processed, battery_ids_for_split, 
                               test_size=args.test_size, validation_size=args.val_size,
                               random_state=42, strategy=args.split_strategy)
    if split_results is None:
        print("Failed to split data.")
        return
    X_train, X_val, X_test, y_train, y_val, y_test = split_results

    if X_train.empty:
        print("Training data is empty after split. Exiting.")
        return
        
    # Imputation
    imputer = SimpleImputer(strategy='median')
    
    X_train_cols, X_train_idx = X_train.columns, X_train.index
    X_val_cols, X_val_idx = (X_val.columns, X_val.index) if not X_val.empty else (X_train.columns, pd.Index([]))
    X_test_cols, X_test_idx = (X_test.columns, X_test.index) if not X_test.empty else (X_train.columns, pd.Index([]))

    X_train_imputed_np = imputer.fit_transform(X_train)
    X_train_imputed = pd.DataFrame(X_train_imputed_np, columns=X_train_cols, index=X_train_idx)
    
    X_val_imputed = pd.DataFrame(columns=X_val_cols, index=X_val_idx, dtype=float)
    if not X_val.empty:
        X_val_imputed_np = imputer.transform(X_val)
        X_val_imputed = pd.DataFrame(X_val_imputed_np, columns=X_val_cols, index=X_val_idx)

    X_test_imputed = pd.DataFrame(columns=X_test_cols, index=X_test_idx, dtype=float)
    if not X_test.empty:
        X_test_imputed_np = imputer.transform(X_test)
        X_test_imputed = pd.DataFrame(X_test_imputed_np, columns=X_test_cols, index=X_test_idx)

    # Feature Scaling (on imputed data)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train_imputed, X_val_imputed, X_test_imputed
    )
    
    scaler_filename = f"scaler_h{args.forecast_horizon}_strat_{args.split_strategy}.joblib"
    joblib.dump(scaler, scaler_filename)
    print(f"\nScaler saved to {scaler_filename}")
    
    imputer_filename = f"imputer_h{args.forecast_horizon}_strat_{args.split_strategy}.joblib"
    joblib.dump(imputer, imputer_filename)
    print(f"Imputer saved to {imputer_filename}")


    model_results, trained_models = train_and_evaluate_models(
        X_train_imputed, y_train,
        X_val_imputed, y_val,
        X_test_imputed, y_test,
        X_train_scaled,  # Pass scaled versions
        X_val_scaled,
        X_test_scaled,
        feature_names=feature_names_list,
        models_to_train=args.models
    )

    print("\n--- Final Results Summary ---")
    for model_name, metrics in model_results.items():
        print(f"\nModel: {model_name.upper()}")
        print(f"  Validation: RMSE={metrics['val_rmse']:.4f}, MAE={metrics['val_mae']:.4f}, R2={metrics['val_r2']:.4f}")
        print(f"  Test:       RMSE={metrics['test_rmse']:.4f}, MAE={metrics['test_mae']:.4f}, R2={metrics['test_r2']:.4f}")

    if args.save_models:
        output_model_dir = "trained_models"
        os.makedirs(output_model_dir, exist_ok=True)
        for model_name, model_instance in trained_models.items():
            model_filename = os.path.join(output_model_dir, f"{model_name}_h{args.forecast_horizon}_strat_{args.split_strategy}.joblib")
            joblib.dump(model_instance, model_filename)
            print(f"Trained model {model_name.upper()} saved to {model_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SOH forecasting models.")
    parser.add_argument("input_file", type=str, help="Path to the processed CSV data file.")
    parser.add_argument("--forecast_horizon", type=int, default=5, help="Forecast horizon for SOH target (e.g., 5 for SOH_target_h5).")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data for the test set.")
    parser.add_argument("--val_size", type=float, default=0.15, help="Proportion of data for the validation set.")
    parser.add_argument("--models", nargs='+', default=['rf', 'gb', 'lr', 'xgb'], help="List of models to train (e.g., rf gb lr xgb). 'gb' is HistGradientBoostingRegressor.")
    parser.add_argument("--save_models", action='store_true', help="Save trained models to disk.")
    parser.add_argument("--split_strategy", type=str, default='chronological', 
                        choices=['chronological', 'group_shuffle_batteries'],
                        help="Data splitting strategy. 'chronological' for forecasting on seen batteries, "
                             "'group_shuffle_batteries' for generalization to unseen batteries.")

    args = parser.parse_args()
    main(args)