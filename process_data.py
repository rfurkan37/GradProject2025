import pandas as pd
import numpy as np
import os
from IPython.display import display # For better display in notebooks

# --- Configuration Constants ---
DATASET_ROOT_DIRECTORY = "battery_alt_dataset" # IMPORTANT: Set this to your dataset path
REGULAR_BATTERIES_FOLDER = "regular_alt_batteries"
RECOMMISSIONED_BATTERIES_FOLDER = "recommissioned_batteries"
SECOND_LIFE_BATTERIES_FOLDER = "second_life_batteries" # Added from paper structure
FOLDERS_TO_PROCESS = [REGULAR_BATTERIES_FOLDER, RECOMMISSIONED_BATTERIES_FOLDER]


COLUMN_MAPPING = {
    "start_time": "start_time", "time": "relative_time_s", "mode": "mode",
    "voltage_charger": "voltage_charger_V", "temperature_battery": "temp_battery_C",
    "voltage_load": "voltage_load_V", "current_load": "current_load_A",
    "temperature_mosfet": "temp_mosfet_C", "temperature_resistor": "temp_resistor_C",
    "mission_type": "mission_type"
}
NOMINAL_CAPACITY_AH = 2.5
SOH_EOL_THRESHOLD_PERCENT = 80.0 # Not used in current processing, but good to keep
MIN_ACCEPTABLE_STABLE_VOLTAGE_V = 4.0 # Min voltage of a 2S pack (2.0V/cell) to start considering data
MINIMUM_VALID_CYCLE_DURATION_S = 60 # Increased for more meaningful cycles
MINIMUM_DATA_POINTS_PER_CYCLE = 10 # Increased for more meaningful cycles
IR_CALCULATION_DURATION_S = 5.0 # Use first 5s for IR calculation
MIN_CURRENT_FOR_IR_CALC_A = 1.0 # Min average current during IR calc period

def load_battery_data(file_path):
    try:
        df = pd.read_csv(file_path, low_memory=False)
        # Rename first to avoid collision if original names are among target names
        df.rename(columns=COLUMN_MAPPING, inplace=True, errors='ignore')

        num_cols = ['relative_time_s', 'voltage_charger_V', 'temp_battery_C', 'voltage_load_V',
                    'current_load_A', 'temp_mosfet_C', 'temp_resistor_C']
        for col in num_cols:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
            else: # Add missing numeric columns with NaN
                print(f"Warning: Numeric column {col} missing in {file_path}. Adding as NaN.")
                df[col] = np.nan

        if 'mode' in df.columns:
            df['mode'] = pd.to_numeric(df['mode'], errors='coerce').fillna(99).astype(int) # 99 for unknown mode
        else:
            print(f"Warning: Mode column missing in {file_path}. Adding as 99.")
            df['mode'] = 99

        if 'mission_type' in df.columns:
            df['mission_type'] = pd.to_numeric(df['mission_type'], errors='coerce').fillna(0).astype(int) # 0 for unknown/default mission
        else:
            print(f"Warning: Mission_type column missing in {file_path}. Adding as 0.")
            df['mission_type'] = 0
            
        # Ensure essential columns for segmentation and basic processing exist
        req_cols_raw = ['relative_time_s', 'mode', 'voltage_load_V', 'current_load_A']
        for r_col in req_cols_raw:
            if r_col not in df.columns:
                print(f"CRITICAL WARNING: Essential raw column {r_col} is missing in {file_path} even after potential rename/add. Cycle processing might fail.")
                # df[r_col] = np.nan # Or handle error more strictly
        return df
    except FileNotFoundError:
        print(f"Error: File not found {file_path}"); return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}"); return None

def segment_discharge_cycles(df, battery_id):
    cycles = []
    if df is None or df.empty or 'mode' not in df.columns: return cycles
    
    # Ensure mode is integer after potential NaN coercion in load
    df['mode'] = pd.to_numeric(df['mode'], errors='coerce').fillna(99).astype(int)

    in_discharge = False
    current_cycle_data = []
    cycle_number = 0 # Per-battery cycle number

    for _, row in df.iterrows():
        if row['mode'] == -1: # Discharge mode
            if not in_discharge:
                in_discharge = True
                current_cycle_data = [] # Start collecting data for a new cycle
            current_cycle_data.append(row)
        elif in_discharge: # Mode changed from -1 (discharge) to something else (0: rest, 1: charge)
            in_discharge = False
            if current_cycle_data:
                cycle_df = pd.DataFrame(current_cycle_data)
                is_ref = False
                # mission_type=0 means reference discharge (as per paper Appendix A.2)
                if 'mission_type' in cycle_df.columns and not cycle_df.empty:
                    mt = pd.to_numeric(cycle_df['mission_type'].iloc[0], errors='coerce')
                    if pd.notna(mt): is_ref = (mt == 0)
                
                cycles.append({
                    "battery_id": battery_id,
                    "cycle_number": cycle_number,
                    "cycle_df": cycle_df,
                    "is_reference": is_ref
                })
                cycle_number += 1
                current_cycle_data = [] # Reset for next potential cycle
    
    # Handle case where file ends during a discharge
    if in_discharge and current_cycle_data:
        cycle_df = pd.DataFrame(current_cycle_data)
        is_ref = False
        if 'mission_type' in cycle_df.columns and not cycle_df.empty:
            mt = pd.to_numeric(cycle_df['mission_type'].iloc[0], errors='coerce')
            if pd.notna(mt): is_ref = (mt == 0)
        cycles.append({
            "battery_id": battery_id,
            "cycle_number": cycle_number,
            "cycle_df": cycle_df,
            "is_reference": is_ref
        })
    return cycles

def filter_invalid_cycles(segmented_cycles_list, battery_id):
    valid_cycles = []
    new_cycle_num = 0 
    for cycle_info in segmented_cycles_list:
        df_raw = cycle_info["cycle_df"] # This is already a DataFrame
        
        # Make a copy for modification to avoid SettingWithCopyWarning on slices later
        df_temp = df_raw.copy()

        crit_cols = ['relative_time_s', 'voltage_load_V', 'current_load_A']
        if any(col not in df_temp.columns for col in crit_cols):
            # print(f"Warning: Battery {battery_id}, Cycle original num (approx) {cycle_info['cycle_number']}: Missing critical columns. Skipping.")
            continue
            
        for col in crit_cols: # Ensure numeric, coerce errors
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
        
        df_temp.dropna(subset=crit_cols, inplace=True) # Drop rows if essential values are NaN
        
        if df_temp.empty or len(df_temp) < MINIMUM_DATA_POINTS_PER_CYCLE:
            # print(f"Warning: Battery {battery_id}, Cycle original num {cycle_info['cycle_number']}: Too few data points after NaNs. Skipping.")
            continue
        
        # Find first point where voltage is sensible for a 2S pack (e.g., > 2.0V * 2 cells = 4.0V)
        # This avoids initial noise or incorrect readings at cycle start.
        first_valid_voltage_idx = df_temp[df_temp['voltage_load_V'] >= MIN_ACCEPTABLE_STABLE_VOLTAGE_V].index.min()

        if pd.isna(first_valid_voltage_idx): # No valid start voltage found
            # print(f"Warning: Battery {battery_id}, Cycle {cycle_info['cycle_number']}: No acceptable start voltage. Skipping.")
            continue
        
        # Trim the cycle DataFrame to start from this valid point
        # Use .loc on the original df_raw to ensure we get all original columns
        df_trimmed = df_raw.loc[first_valid_voltage_idx:].copy() 
        
        # Re-check numeric and drop NaNs for crit_cols on the trimmed version
        for col in crit_cols: df_trimmed[col] = pd.to_numeric(df_trimmed[col], errors='coerce')
        df_trimmed.dropna(subset=crit_cols, inplace=True)

        if df_trimmed.empty or len(df_trimmed) < MINIMUM_DATA_POINTS_PER_CYCLE:
            # print(f"Warning: Battery {battery_id}, Cycle {cycle_info['cycle_number']}: Too few points after trimming/NaN drop. Skipping.")
            continue
        
        # Check duration of the trimmed, cleaned cycle
        time_trimmed_numeric = pd.to_numeric(df_trimmed['relative_time_s'], errors='coerce').dropna()
        if len(time_trimmed_numeric) < 2: continue # Need at least two points for duration
        
        duration_trimmed = time_trimmed_numeric.iloc[-1] - time_trimmed_numeric.iloc[0]
        if duration_trimmed < MINIMUM_VALID_CYCLE_DURATION_S:
            # print(f"Warning: Battery {battery_id}, Cycle {cycle_info['cycle_number']}: Duration too short after trim. Skipping.")
            continue
        
        updated_cycle_info = cycle_info.copy()
        updated_cycle_info["cycle_df"] = df_trimmed # This is the cleaned and trimmed DataFrame
        updated_cycle_info["cycle_number"] = new_cycle_num # Renumber valid cycles
        valid_cycles.append(updated_cycle_info)
        new_cycle_num +=1
    return valid_cycles

def extract_cycle_features(cycle_info):
    cycle_df_original = cycle_info["cycle_df"] # This is already the filtered, trimmed DF
    features = {
        "battery_id": cycle_info["battery_id"],
        "cycle_number": cycle_info["cycle_number"], # This is the re-sequenced valid cycle number
        "is_reference_cycle": cycle_info["is_reference"],
        "internal_resistance_ohm": np.nan # Initialize
    }
    
    # Make a copy to safely perform numeric conversions and drops
    df_cleaned = cycle_df_original.copy()

    # Ensure essential columns are numeric, coercing errors to NaN
    std_cols_for_calc = ['relative_time_s', 'voltage_load_V', 'current_load_A', 'temp_battery_C']
    for col in std_cols_for_calc:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        else: # Should not happen if filter_invalid_cycles and load_battery_data work
            # print(f"Dev Warning: Column {col} unexpectedly missing in extract_cycle_features for {features['battery_id']} C{features['cycle_number']}")
            df_cleaned[col] = np.nan 
            
    # Drop rows where these specific calculation-critical columns are NaN
    df_cleaned.dropna(subset=['relative_time_s', 'voltage_load_V', 'current_load_A'], inplace=True)

    if df_cleaned.empty or len(df_cleaned) < 2: # Need at least 2 points for duration/diff
        # print(f"Warning: Not enough valid data points for feature extraction in cycle {cycle_info['cycle_number']} for battery {cycle_info['battery_id']} after final clean.")
        return features # Return basic features with NaNs for calculated ones

    time_s_vals = df_cleaned['relative_time_s'].values
    current_A_vals = df_cleaned['current_load_A'].values
    voltage_V_vals = df_cleaned['voltage_load_V'].values

    features['discharge_duration_s'] = time_s_vals[-1] - time_s_vals[0]
    
    if len(time_s_vals) > 1:
        delta_t_intervals = np.diff(time_s_vals) 
        avg_current_intervals = (current_A_vals[:-1] + current_A_vals[1:]) / 2.0
        features['capacity_Ah'] = np.sum(avg_current_intervals * delta_t_intervals) / 3600.0

        power_W_vals = voltage_V_vals * current_A_vals
        avg_power_intervals = (power_W_vals[:-1] + power_W_vals[1:]) / 2.0
        features['energy_Wh'] = np.sum(avg_power_intervals * delta_t_intervals) / 3600.0
    else:
        features['capacity_Ah'] = 0.0
        features['energy_Wh'] = 0.0

    features['avg_current_A'] = np.mean(current_A_vals) 
    features['avg_voltage_V'] = np.mean(voltage_V_vals) 
    features['start_voltage_V'] = voltage_V_vals[0]
    features['end_voltage_V'] = voltage_V_vals[-1]
    features['delta_voltage_V'] = voltage_V_vals[0] - voltage_V_vals[-1] # Positive for discharge
    
    if 'temp_battery_C' in df_cleaned.columns and df_cleaned['temp_battery_C'].notna().any():
        temp_C_vals = df_cleaned['temp_battery_C'].dropna().values
        if temp_C_vals.size > 0:
            features['avg_temp_C'] = np.mean(temp_C_vals)
            features['start_temp_C'] = temp_C_vals[0] if temp_C_vals.size > 0 else np.nan
            features['end_temp_C'] = temp_C_vals[-1] if temp_C_vals.size > 0 else np.nan
            features['delta_temp_C'] = (temp_C_vals[-1] - temp_C_vals[0]) if temp_C_vals.size > 1 else 0.0
            features['max_temp_C'] = np.max(temp_C_vals)
        else:
             for k_temp in ['avg_temp_C', 'start_temp_C', 'end_temp_C', 'delta_temp_C', 'max_temp_C']: features[k_temp] = np.nan
    else:
        for k_temp in ['avg_temp_C', 'start_temp_C', 'end_temp_C', 'delta_temp_C', 'max_temp_C']: features[k_temp] = np.nan

    features['avg_power_W'] = np.mean(voltage_V_vals * current_A_vals)

    # --- Internal Resistance Calculation ---
    if len(df_cleaned) > 1:
        # Time relative to the start of this cleaned cycle segment
        elapsed_time_in_cleaned_cycle = time_s_vals - time_s_vals[0]
        
        # Find points within the IR_CALCULATION_DURATION_S
        ir_segment_mask = (elapsed_time_in_cleaned_cycle <= IR_CALCULATION_DURATION_S) & \
                          (elapsed_time_in_cleaned_cycle >= 0) # Ensure positive time
        
        # Data points for IR calculation
        v_ir_segment = voltage_V_vals[ir_segment_mask]
        i_ir_segment = current_A_vals[ir_segment_mask]

        if len(v_ir_segment) > 1 and len(i_ir_segment) > 1: # Need at least two points in the segment
            v_start_ir_calc = v_ir_segment[0]
            v_end_ir_calc = v_ir_segment[-1]
            
            # Average current over this specific IR calculation segment
            i_avg_ir_calc_segment = np.mean(i_ir_segment)

            if pd.notna(i_avg_ir_calc_segment) and abs(i_avg_ir_calc_segment) >= MIN_CURRENT_FOR_IR_CALC_A:
                delta_v_ir_segment = v_start_ir_calc - v_end_ir_calc # Should be positive
                if delta_v_ir_segment >= 0 and i_avg_ir_calc_segment != 0:
                    features['internal_resistance_ohm'] = delta_v_ir_segment / i_avg_ir_calc_segment
    return features

def calculate_q_initial_and_soh(all_cycles_df_for_battery, nominal_capacity_ah=NOMINAL_CAPACITY_AH):
    df = all_cycles_df_for_battery.copy()
    q_initial = nominal_capacity_ah # Default q_initial
    
    # Try to find the capacity of the first valid reference cycle
    ref_q_cycles = df[df['is_reference_cycle'].fillna(False) & df['capacity_Ah'].notna() & (df['capacity_Ah'] > 0)]
    
    if not ref_q_cycles.empty:
        # Sort by cycle number to ensure we get the earliest reference cycle
        first_ref_cap = ref_q_cycles.sort_values('cycle_number')['capacity_Ah'].iloc[0]
        q_initial = first_ref_cap
    else: # Fallback to first overall valid cycle if no valid reference cycle capacity found
        overall_q_cycles = df[df['capacity_Ah'].notna() & (df['capacity_Ah'] > 0)]
        if not overall_q_cycles.empty:
            first_overall_cap = overall_q_cycles.sort_values('cycle_number')['capacity_Ah'].iloc[0]
            q_initial = first_overall_cap
            # print(f"Battery {df['battery_id'].iloc[0]}: No valid reference cycle for Q_initial. Used first overall cycle capacity: {q_initial:.2f} Ah.")
        # else:
            # print(f"Battery {df['battery_id'].iloc[0]}: No valid cycles with capacity > 0 found for Q_initial. Using nominal: {nominal_capacity_ah} Ah.")
            
    df['q_initial_Ah'] = q_initial
    if q_initial > 0 and 'capacity_Ah' in df.columns:
        df['SOH_cycle_capacity_%'] = (df['capacity_Ah'] / q_initial) * 100.0
    else:
        df['SOH_cycle_capacity_%'] = np.nan
    return df

def add_reference_context_features(all_cycles_df_for_battery_with_soh):
    df = all_cycles_df_for_battery_with_soh.sort_values('cycle_number').copy()
    df['cycles_since_last_ref'] = 0
    df['SOH_at_last_ref_%'] = np.nan # SOH based on capacity, from the last reference cycle
    
    last_ref_cycle_num = -1
    soh_val_at_last_ref = np.nan

    for i, row in df.iterrows(): # Use iterrows for simplicity here, performance not critical for this part
        if row['is_reference_cycle']:
            df.loc[i, 'cycles_since_last_ref'] = 0
            last_ref_cycle_num = row['cycle_number']
            if pd.notna(row['SOH_cycle_capacity_%']):
                soh_val_at_last_ref = row['SOH_cycle_capacity_%']
            df.loc[i, 'SOH_at_last_ref_%'] = soh_val_at_last_ref # SOH *of this* reference cycle
        else:
            if last_ref_cycle_num != -1: # A reference cycle has occurred
                df.loc[i, 'cycles_since_last_ref'] = row['cycle_number'] - last_ref_cycle_num
                df.loc[i, 'SOH_at_last_ref_%'] = soh_val_at_last_ref # SOH from the *previous* ref cycle
            else: # Before the first reference cycle
                df.loc[i, 'cycles_since_last_ref'] = row['cycle_number'] + 1 
                df.loc[i, 'SOH_at_last_ref_%'] = np.nan 
    return df

def add_health_indicators(all_cycles_df_for_battery):
    df = all_cycles_df_for_battery.copy()
    # Ensure numeric, though should be by now
    for col in ['start_voltage_V', 'end_voltage_V', 'capacity_Ah']:
        if col not in df.columns: df[col] = np.nan
        else: df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1. Voltage Ratio (captures relative drop)
    df['HI_voltage_ratio_end_start'] = np.where(
        df['start_voltage_V'].fillna(0) != 0,
        df['end_voltage_V'] / df['start_voltage_V'], np.nan
    )

    # 2. Capacity fade from previous cycle (absolute Ah)
    # This is a lag-1 difference of capacity_Ah.
    # It's often useful to have directly.
    df['HI_capacity_fade_from_prev_cycle_Ah'] = df['capacity_Ah'].diff()
    
    # 3. Internal Resistance change from previous cycle (Ohms)
    # Requires 'internal_resistance_ohm' to be calculated first.
    if 'internal_resistance_ohm' in df.columns:
        df['internal_resistance_ohm'] = pd.to_numeric(df['internal_resistance_ohm'], errors='coerce')
        df['HI_IR_change_from_prev_cycle_ohm'] = df['internal_resistance_ohm'].diff()
    else:
        df['HI_IR_change_from_prev_cycle_ohm'] = np.nan
        
    # Consider adding other simple, academically recognized HIs if needed,
    # e.g., related to specific points on the discharge curve if you had more granular data.
    return df

def add_future_soh_target(all_batteries_df, soh_col_name='SOH_cycle_capacity_%', forecast_horizon=1):
    if soh_col_name not in all_batteries_df.columns:
        all_batteries_df[f'SOH_target_h{forecast_horizon}'] = np.nan
        return all_batteries_df
    df = all_batteries_df.copy()
    df = df.sort_values(['battery_id', 'cycle_number']) 
    df[f'SOH_target_h{forecast_horizon}'] = df.groupby('battery_id')[soh_col_name].shift(-forecast_horizon)
    return df

def add_lag_features(df_master, features_to_lag, lags):
    df = df_master.copy()
    df = df.sort_values(['battery_id', 'cycle_number'])
    for feature in features_to_lag:
        if feature in df.columns:
            for lag in lags:
                df[f'{feature}_lag{lag}'] = df.groupby('battery_id')[feature].shift(lag)
        else:
            print(f"Warning: Feature {feature} not found for lag feature creation.")
    return df

def add_rolling_stats_features(df_master, features_for_rolling, windows, stats=['mean']): # Default to mean only
    df = df_master.copy()
    df = df.sort_values(['battery_id', 'cycle_number'])
    for feature in features_for_rolling:
        if feature in df.columns:
            for window in windows:
                if 'mean' in stats:
                    df[f'{feature}_roll_mean_w{window}'] = df.groupby('battery_id')[feature].rolling(window=window, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
                if 'std' in stats and window > 1: # Std dev needs at least 2 points
                    df[f'{feature}_roll_std_w{window}'] = df.groupby('battery_id')[feature].rolling(window=window, min_periods=2).std().shift(1).reset_index(level=0, drop=True)
        else:
            print(f"Warning: Feature {feature} not found for rolling stats creation.")
    return df

def process_battery_dataset(root_dir, folders_list, single_battery_id=None, forecast_horizon_for_target=1, add_rolling_features=True):
    all_batteries_processed_list = []
    found_single = False

    for folder_name in folders_list:
        if found_single and single_battery_id: break
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder {folder_path} not found. Skipping.")
            continue
        
        for file_name in sorted(os.listdir(folder_path)): # Sort for consistent processing order
            if file_name.endswith(".csv"):
                current_batt_id = os.path.splitext(file_name)[0]
                # Convert X.Y style IDs to string "X_Y" for easier handling if needed later (e.g. column names)
                # current_batt_id = current_batt_id.replace('.', '_') 

                if single_battery_id and current_batt_id != single_battery_id: continue
                
                if not single_battery_id or (single_battery_id and not found_single) :
                    print(f"\nProcessing battery: {current_batt_id} from folder {folder_name}")
                found_single = True
                
                raw_df = load_battery_data(os.path.join(folder_path, file_name))
                if raw_df is None or raw_df.empty: print(f"  No data loaded for {current_batt_id}."); continue
                
                segmented = segment_discharge_cycles(raw_df, current_batt_id)
                if not segmented: print(f"  No discharge cycles segmented for {current_batt_id}."); continue
                
                valid_cycles_info = filter_invalid_cycles(segmented, current_batt_id)
                if not valid_cycles_info: print(f"  No valid cycles after filtering for {current_batt_id}."); continue
                
                battery_all_cycles_features_list = []
                for info in valid_cycles_info:
                    features = extract_cycle_features(info)
                    # Only add if core features like capacity could be calculated (proxy for a good cycle)
                    if 'capacity_Ah' in features and pd.notna(features['capacity_Ah']):
                         battery_all_cycles_features_list.append(features)
                
                if not battery_all_cycles_features_list: print(f"  No features extracted from valid cycles for {current_batt_id}."); continue
                
                df_all_cycles_for_batt = pd.DataFrame(battery_all_cycles_features_list)
                df_all_cycles_for_batt = df_all_cycles_for_batt.sort_values('cycle_number').reset_index(drop=True)

                df_with_soh = calculate_q_initial_and_soh(df_all_cycles_for_batt)
                df_with_ref_context = add_reference_context_features(df_with_soh)
                df_enriched_with_hi = add_health_indicators(df_with_ref_context) # HI_capacity_fade relies on .diff() so call on per-battery df
                
                all_batteries_processed_list.append(df_enriched_with_hi)
                if single_battery_id: break
        if single_battery_id and found_single: break
                        
    if not all_batteries_processed_list:
        print("No battery data successfully processed into list.")
        return pd.DataFrame()
        
    master_df = pd.concat(all_batteries_processed_list, ignore_index=True)
    master_df = master_df.sort_values(['battery_id', 'cycle_number']).reset_index(drop=True)

    # --- Add Lag and Rolling Features ---
    print("\nAdding selected lag and rolling features...")
    
    # Primary features for detailed history
    features_for_sL_sR_detailed = ['SOH_cycle_capacity_%', 'internal_resistance_ohm']
    lags_detailed = [1, 2] # More recent lags
    roll_windows_detailed = [3] # Short-term smoothing

    # Secondary features for general trend
    features_for_sL_sR_trend = ['capacity_Ah', 'delta_voltage_V', 'max_temp_C']
    lags_trend = [1, 3] # One recent, one slightly further
    # No rolling for these initially to keep feature count down, simple lags are enough for trend indication.

    # Create lag features
    df_with_lags = add_lag_features(master_df, features_for_sL_sR_detailed, lags_detailed)
    df_with_lags = add_lag_features(df_with_lags, features_for_sL_sR_trend, lags_trend)
    
    master_df_featured = df_with_lags
    
    # Create rolling features (only for detailed set) if enabled
    if add_rolling_features:
        print("Adding selected rolling features...")
        # Important: Pass the DataFrame that already has the lags needed for calculation if shift(1) is used,
        # or ensure rolling is applied correctly based on original values.
        # The current add_rolling_stats_features uses .shift(1) so it's using past values relative to current cycle.
        master_df_featured = add_rolling_stats_features(df_with_lags, features_for_sL_sR_detailed, roll_windows_detailed)

    # --- Add Future SOH Target ---
    print(f"Adding future SOH target with horizon {forecast_horizon_for_target}...")
    master_df_with_target = add_future_soh_target(master_df_featured, # Use the featured df
                                                  soh_col_name='SOH_cycle_capacity_%', 
                                                  forecast_horizon=forecast_horizon_for_target)
    
    return master_df_with_target


# --- Main Execution ---
if not os.path.exists(DATASET_ROOT_DIRECTORY):
     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
     print(f"!!! DATASET_ROOT_DIRECTORY '{DATASET_ROOT_DIRECTORY}' not found. Please set it. !!!")
     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
     master_cycle_df = pd.DataFrame()
else:
    PROCESS_SINGLE_BATTERY_ID = None 
    FORECAST_HORIZON = 5
    ADD_ROLLING_FEATURES = False
    
    # Define the lag configuration that process_battery_dataset's internal logic uses
    # This is for the display logic at the end to correctly estimate rows to skip.
    LAGS_CONFIG_USED_FOR_PROCESSING = [1, 2, 3, 5] 

    if PROCESS_SINGLE_BATTERY_ID:
        print(f"Attempting to process single battery: {PROCESS_SINGLE_BATTERY_ID}")
    else:
        print(f"Starting battery data processing for all batteries from: {DATASET_ROOT_DIRECTORY}")
        
    print(f"Rolling features: {'Enabled' if ADD_ROLLING_FEATURES else 'Disabled'}")
    
    master_cycle_df = process_battery_dataset(
        DATASET_ROOT_DIRECTORY,
        FOLDERS_TO_PROCESS,
        single_battery_id=PROCESS_SINGLE_BATTERY_ID,
        forecast_horizon_for_target=FORECAST_HORIZON,
        add_rolling_features=ADD_ROLLING_FEATURES
        # Note: process_battery_dataset uses its own internal definition for lags/windows passed to add_lag_features.
        # The LAGS_CONFIG_USED_FOR_PROCESSING here is for the display logic below.
    )

    if master_cycle_df.empty:
        if PROCESS_SINGLE_BATTERY_ID:
            print(f"\nNo data was processed for battery '{PROCESS_SINGLE_BATTERY_ID}'. Check ID, file names, and logs.")
        else:
            print("\nNo data was processed for any battery. Check logs.")
    else:
        print("\n\n--- Master DataFrame Head (with Lag/Roll Features & Target) ---")
        target_col_name = f'SOH_target_h{FORECAST_HORIZON}'
        
        display_cols_subset = [
            'battery_id', 'cycle_number', 'SOH_cycle_capacity_%', target_col_name,
            'internal_resistance_ohm', 'internal_resistance_ohm_lag1',
            'capacity_Ah_roll_mean_w3', 'max_temp_C_lag1', 'is_reference_cycle',
            'cycles_since_last_ref', 'SOH_at_last_ref_%'
        ]
        display_cols_present = [col for col in display_cols_subset if col in master_cycle_df.columns]
        
        if not master_cycle_df.empty:
            first_batt_id_in_master = master_cycle_df['battery_id'].iloc[0]
            print(f"\nDisplaying some columns for battery: {first_batt_id_in_master}")
            
            display_df_sample = master_cycle_df[master_cycle_df['battery_id'] == first_batt_id_in_master]
            
            # Determine rows to skip based on the max lag used during processing
            skip_rows_for_display = 0
            if LAGS_CONFIG_USED_FOR_PROCESSING: # Ensure the list is not empty
                skip_rows_for_display = max(LAGS_CONFIG_USED_FOR_PROCESSING)
            
            # Show some rows where lag/roll features might be populated and target is not NaN
            # Adjust starting index based on max lag to see populated lag features
            start_index_for_display = skip_rows_for_display + 2 
            end_index_for_display = start_index_for_display + FORECAST_HORIZON + 8 # Show a few rows with target

            if len(display_df_sample) > start_index_for_display :
                 display(display_df_sample[display_cols_present].iloc[start_index_for_display : end_index_for_display])
            else: # Fallback if not enough rows after skipping for lags
                 display(display_df_sample[display_cols_present].head(10 + FORECAST_HORIZON))

        print(f"\nMaster DataFrame shape: {master_cycle_df.shape}")
        unique_batteries = master_cycle_df['battery_id'].nunique()
        print(f"Unique batteries in final DataFrame: {unique_batteries}")
        
        if target_col_name in master_cycle_df.columns:
            print(f"Number of non-NaN future SOH targets ({target_col_name}): {master_cycle_df[target_col_name].notna().sum()}")
            print(f"Number of NaN future SOH targets: {master_cycle_df[target_col_name].isna().sum()}")
        
        print("\nProcessing complete. DataFrame 'master_cycle_df' includes lag/roll features and target.")
        
        output_filename = f"processed_battery_data_lag_roll_target_h{FORECAST_HORIZON}"
        if PROCESS_SINGLE_BATTERY_ID:
            output_filename += f"_{PROCESS_SINGLE_BATTERY_ID.replace('.', '_')}"
        else:
            output_filename += "_ALL"
        output_filename += ".csv"
        
        master_cycle_df.to_csv(output_filename, index=False)
        print(f"DataFrame saved to {output_filename}")