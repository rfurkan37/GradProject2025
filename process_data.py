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
SOH_EOL_THRESHOLD_PERCENT = 80.0 
MIN_ACCEPTABLE_STABLE_VOLTAGE_V = 4.0 
MINIMUM_VALID_CYCLE_DURATION_S = 60 
MINIMUM_DATA_POINTS_PER_CYCLE = 10 
IR_CALCULATION_DURATION_S = 5.0 
MIN_CURRENT_FOR_IR_CALC_A = 0.5 # Lowered from 1.0

def load_battery_data(file_path):
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df.rename(columns=COLUMN_MAPPING, inplace=True, errors='ignore')

        num_cols = ['relative_time_s', 'voltage_charger_V', 'temp_battery_C', 'voltage_load_V',
                    'current_load_A', 'temp_mosfet_C', 'temp_resistor_C']
        for col in num_cols:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
            else: 
                # print(f"Warning: Numeric column {col} missing in {file_path}. Adding as NaN.") # Reduced verbosity
                df[col] = np.nan

        if 'mode' in df.columns:
            df['mode'] = pd.to_numeric(df['mode'], errors='coerce').fillna(99).astype(int) 
        else:
            # print(f"Warning: Mode column missing in {file_path}. Adding as 99.") # Reduced verbosity
            df['mode'] = 99

        if 'mission_type' in df.columns:
            df['mission_type'] = pd.to_numeric(df['mission_type'], errors='coerce').fillna(0).astype(int) 
        else:
            # print(f"Warning: Mission_type column missing in {file_path}. Adding as 0.") # Reduced verbosity
            df['mission_type'] = 0
            
        req_cols_raw = ['relative_time_s', 'mode', 'voltage_load_V', 'current_load_A']
        for r_col in req_cols_raw:
            if r_col not in df.columns:
                print(f"CRITICAL WARNING: Essential raw column {r_col} is missing in {file_path} even after potential rename/add. Cycle processing might fail.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found {file_path}"); return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}"); return None

def segment_discharge_cycles(df, battery_id):
    cycles = []
    if df is None or df.empty or 'mode' not in df.columns: return cycles
    
    df['mode'] = pd.to_numeric(df['mode'], errors='coerce').fillna(99).astype(int)

    in_discharge = False
    current_cycle_data = []
    cycle_number = 0 

    for _, row in df.iterrows():
        if row['mode'] == -1: 
            if not in_discharge:
                in_discharge = True
                current_cycle_data = [] 
            current_cycle_data.append(row)
        elif in_discharge: 
            in_discharge = False
            if current_cycle_data:
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
                cycle_number += 1
                current_cycle_data = [] 
    
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
        df_raw = cycle_info["cycle_df"] 
        df_temp = df_raw.copy()

        crit_cols = ['relative_time_s', 'voltage_load_V', 'current_load_A']
        if any(col not in df_temp.columns for col in crit_cols):
            continue
            
        for col in crit_cols: 
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
        
        df_temp.dropna(subset=crit_cols, inplace=True) 
        
        if df_temp.empty or len(df_temp) < MINIMUM_DATA_POINTS_PER_CYCLE:
            continue
        
        first_valid_voltage_idx = df_temp[df_temp['voltage_load_V'] >= MIN_ACCEPTABLE_STABLE_VOLTAGE_V].index.min()

        if pd.isna(first_valid_voltage_idx): 
            continue
        
        df_trimmed = df_raw.loc[first_valid_voltage_idx:].copy() 
        
        for col in crit_cols: df_trimmed[col] = pd.to_numeric(df_trimmed[col], errors='coerce')
        df_trimmed.dropna(subset=crit_cols, inplace=True)

        if df_trimmed.empty or len(df_trimmed) < MINIMUM_DATA_POINTS_PER_CYCLE:
            continue
        
        time_trimmed_numeric = pd.to_numeric(df_trimmed['relative_time_s'], errors='coerce').dropna()
        if len(time_trimmed_numeric) < 2: continue 
        
        duration_trimmed = time_trimmed_numeric.iloc[-1] - time_trimmed_numeric.iloc[0]
        if duration_trimmed < MINIMUM_VALID_CYCLE_DURATION_S:
            continue
        
        updated_cycle_info = cycle_info.copy()
        updated_cycle_info["cycle_df"] = df_trimmed 
        updated_cycle_info["cycle_number"] = new_cycle_num 
        valid_cycles.append(updated_cycle_info)
        new_cycle_num +=1
    return valid_cycles

def extract_cycle_features(cycle_info):
    cycle_df_original = cycle_info["cycle_df"] 
    features = {
        "battery_id": cycle_info["battery_id"],
        "cycle_number": cycle_info["cycle_number"], 
        "is_reference_cycle": cycle_info["is_reference"],
        "internal_resistance_ohm": np.nan 
    }
    
    df_cleaned = cycle_df_original.copy()

    std_cols_for_calc = ['relative_time_s', 'voltage_load_V', 'current_load_A', 'temp_battery_C']
    for col in std_cols_for_calc:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        else: 
            df_cleaned[col] = np.nan 
            
    df_cleaned.dropna(subset=['relative_time_s', 'voltage_load_V', 'current_load_A'], inplace=True)

    if df_cleaned.empty or len(df_cleaned) < 2: 
        return features 

    time_s_vals = df_cleaned['relative_time_s'].values
    current_A_vals = df_cleaned['current_load_A'].values
    voltage_V_vals = df_cleaned['voltage_load_V'].values
    
    # Add mission_type as a feature if present
    if 'mission_type' in df_cleaned.columns and not df_cleaned['mission_type'].empty:
        # Assuming mission_type is constant for the cycle, take the first value
        features['mission_type'] = pd.to_numeric(df_cleaned['mission_type'].iloc[0], errors='coerce')
    else:
        features['mission_type'] = 0 # Default or NaN if preferred

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
    features['delta_voltage_V'] = voltage_V_vals[0] - voltage_V_vals[-1] 
    
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

    if len(df_cleaned) > 1:
        elapsed_time_in_cleaned_cycle = time_s_vals - time_s_vals[0]
        ir_segment_mask = (elapsed_time_in_cleaned_cycle <= IR_CALCULATION_DURATION_S) & \
                          (elapsed_time_in_cleaned_cycle >= 0) 
        v_ir_segment = voltage_V_vals[ir_segment_mask]
        i_ir_segment = current_A_vals[ir_segment_mask]

        if len(v_ir_segment) > 1 and len(i_ir_segment) > 1: 
            v_start_ir_calc = v_ir_segment[0]
            v_end_ir_calc = v_ir_segment[-1]
            i_avg_ir_calc_segment = np.mean(i_ir_segment)

            if pd.notna(i_avg_ir_calc_segment) and abs(i_avg_ir_calc_segment) >= MIN_CURRENT_FOR_IR_CALC_A:
                delta_v_ir_segment = v_start_ir_calc - v_end_ir_calc 
                if delta_v_ir_segment >= 0 and i_avg_ir_calc_segment != 0: # Avoid division by zero or negative resistance
                    features['internal_resistance_ohm'] = delta_v_ir_segment / i_avg_ir_calc_segment
    return features

def calculate_q_initial_and_soh(all_cycles_df_for_battery, nominal_capacity_ah=NOMINAL_CAPACITY_AH):
    df = all_cycles_df_for_battery.copy()
    q_initial = nominal_capacity_ah 
    
    ref_q_cycles = df[df['is_reference_cycle'].fillna(False) & df['capacity_Ah'].notna() & (df['capacity_Ah'] > 0)]
    
    if not ref_q_cycles.empty:
        first_ref_cap = ref_q_cycles.sort_values('cycle_number')['capacity_Ah'].iloc[0]
        q_initial = first_ref_cap
    else: 
        overall_q_cycles = df[df['capacity_Ah'].notna() & (df['capacity_Ah'] > 0)]
        if not overall_q_cycles.empty:
            first_overall_cap = overall_q_cycles.sort_values('cycle_number')['capacity_Ah'].iloc[0]
            q_initial = first_overall_cap
            
    df['q_initial_Ah'] = q_initial
    if q_initial > 0 and 'capacity_Ah' in df.columns:
        df['SOH_cycle_capacity_%'] = (df['capacity_Ah'] / q_initial) * 100.0
    else:
        df['SOH_cycle_capacity_%'] = np.nan
    return df

def add_reference_context_features(all_cycles_df_for_battery_with_soh):
    df = all_cycles_df_for_battery_with_soh.sort_values('cycle_number').copy()
    df['cycles_since_last_ref'] = 0
    df['SOH_at_last_ref_%'] = np.nan 
    
    last_ref_cycle_num = -1
    soh_val_at_last_ref = np.nan

    for i, row in df.iterrows(): 
        if row['is_reference_cycle']:
            df.loc[i, 'cycles_since_last_ref'] = 0
            last_ref_cycle_num = row['cycle_number']
            if pd.notna(row['SOH_cycle_capacity_%']):
                soh_val_at_last_ref = row['SOH_cycle_capacity_%']
            df.loc[i, 'SOH_at_last_ref_%'] = soh_val_at_last_ref 
        else:
            if last_ref_cycle_num != -1: 
                df.loc[i, 'cycles_since_last_ref'] = row['cycle_number'] - last_ref_cycle_num
                df.loc[i, 'SOH_at_last_ref_%'] = soh_val_at_last_ref 
            else: 
                df.loc[i, 'cycles_since_last_ref'] = row['cycle_number'] + 1 
                df.loc[i, 'SOH_at_last_ref_%'] = np.nan 
    return df

def add_health_indicators(all_cycles_df_for_battery):
    df = all_cycles_df_for_battery.copy()
    for col in ['start_voltage_V', 'end_voltage_V', 'capacity_Ah']:
        if col not in df.columns: df[col] = np.nan
        else: df[col] = pd.to_numeric(df[col], errors='coerce')

    df['HI_voltage_ratio_end_start'] = np.where(
        df['start_voltage_V'].fillna(0) != 0,
        df['end_voltage_V'] / df['start_voltage_V'], np.nan
    )
    df['HI_capacity_fade_from_prev_cycle_Ah'] = df['capacity_Ah'].diff()
    
    if 'internal_resistance_ohm' in df.columns:
        df['internal_resistance_ohm'] = pd.to_numeric(df['internal_resistance_ohm'], errors='coerce')
        df['HI_IR_change_from_prev_cycle_ohm'] = df['internal_resistance_ohm'].diff()
    else:
        df['HI_IR_change_from_prev_cycle_ohm'] = np.nan
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

def add_rolling_stats_features(df_master, features_for_rolling, windows, stats=['mean', 'std']): # Added std
    df = df_master.copy()
    df = df.sort_values(['battery_id', 'cycle_number'])
    for feature in features_for_rolling:
        if feature in df.columns:
            for window in windows:
                if 'mean' in stats:
                    df[f'{feature}_roll_mean_w{window}'] = df.groupby('battery_id')[feature].rolling(window=window, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
                if 'std' in stats and window > 1: 
                    df[f'{feature}_roll_std_w{window}'] = df.groupby('battery_id')[feature].rolling(window=window, min_periods=max(2,int(window/2))).std().shift(1).reset_index(level=0, drop=True) # min_periods adjusted for std
        else:
            print(f"Warning: Feature {feature} not found for rolling stats creation.")
    return df

def process_battery_dataset(root_dir, folders_list, single_battery_id=None, forecast_horizon_for_target=1, add_rolling_features_flag=True):
    all_batteries_processed_list = []
    found_single = False

    for folder_name in folders_list:
        if found_single and single_battery_id: break
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder {folder_path} not found. Skipping.")
            continue
        
        for file_name in sorted(os.listdir(folder_path)): 
            if file_name.endswith(".csv"):
                current_batt_id = os.path.splitext(file_name)[0]
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
                    if 'capacity_Ah' in features and pd.notna(features['capacity_Ah']):
                         battery_all_cycles_features_list.append(features)
                
                if not battery_all_cycles_features_list: print(f"  No features extracted from valid cycles for {current_batt_id}."); continue
                
                df_all_cycles_for_batt = pd.DataFrame(battery_all_cycles_features_list)
                df_all_cycles_for_batt = df_all_cycles_for_batt.sort_values('cycle_number').reset_index(drop=True)

                df_with_soh = calculate_q_initial_and_soh(df_all_cycles_for_batt)
                df_with_ref_context = add_reference_context_features(df_with_soh)
                df_enriched_with_hi = add_health_indicators(df_with_ref_context) 
                
                all_batteries_processed_list.append(df_enriched_with_hi)
                if single_battery_id: break
        if single_battery_id and found_single: break
                        
    if not all_batteries_processed_list:
        print("No battery data successfully processed into list.")
        return pd.DataFrame()
        
    master_df = pd.concat(all_batteries_processed_list, ignore_index=True)
    master_df = master_df.sort_values(['battery_id', 'cycle_number']).reset_index(drop=True)

    print("\nAdding selected lag and rolling features...")
    
    features_for_sL_sR_detailed = ['SOH_cycle_capacity_%', 'internal_resistance_ohm', 'avg_temp_C', 'delta_voltage_V'] # Expanded a bit
    lags_detailed = [1, 2, 5, 10] # Expanded lags
    roll_windows_detailed = [3, 5, 10] # Expanded windows, std will be calculated too

    features_for_sL_sR_trend = ['capacity_Ah', 'max_temp_C', 'discharge_duration_s'] # Slightly adjusted
    lags_trend = [1, 3, 5] # Expanded lags

    df_with_lags = add_lag_features(master_df, features_for_sL_sR_detailed, lags_detailed)
    df_with_lags = add_lag_features(df_with_lags, features_for_sL_sR_trend, lags_trend)
    
    master_df_featured = df_with_lags
    
    if add_rolling_features_flag: # Use the passed flag
        print("Adding selected rolling features (mean and std)...")
        master_df_featured = add_rolling_stats_features(df_with_lags, features_for_sL_sR_detailed, roll_windows_detailed, stats=['mean', 'std'])

    print(f"Adding future SOH target with horizon {forecast_horizon_for_target}...")
    master_df_with_target = add_future_soh_target(master_df_featured, 
                                                  soh_col_name='SOH_cycle_capacity_%', 
                                                  forecast_horizon=forecast_horizon_for_target)
    
    return master_df_with_target


# --- Main Execution ---
if __name__ == "__main__": # Ensure this runs only when script is executed
    if not os.path.exists(DATASET_ROOT_DIRECTORY):
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print(f"!!! DATASET_ROOT_DIRECTORY '{DATASET_ROOT_DIRECTORY}' not found. Please set it. !!!")
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         master_cycle_df = pd.DataFrame()
    else:
        PROCESS_SINGLE_BATTERY_ID = None 
        FORECAST_HORIZON = 5
        ADD_ROLLING_FEATURES_MAIN = True # Changed to True by default
        
        # Updated for display logic, reflecting new config in process_battery_dataset
        # Max of ([1,2,5,10] from detailed, [1,3,5] from trend) = 10
        # Max of rolling windows ([3,5,10]) with shift(1) means effectively lag of window size.
        # So max overall "lookback" for NaNs is roughly max_lag or max_window.
        MAX_LOOKBACK_FOR_NAN_DISPLAY = 10 

        if PROCESS_SINGLE_BATTERY_ID:
            print(f"Attempting to process single battery: {PROCESS_SINGLE_BATTERY_ID}")
        else:
            print(f"Starting battery data processing for all batteries from: {DATASET_ROOT_DIRECTORY}")
            
        print(f"Rolling features generation: {'Enabled' if ADD_ROLLING_FEATURES_MAIN else 'Disabled'}")
        
        master_cycle_df = process_battery_dataset(
            DATASET_ROOT_DIRECTORY,
            FOLDERS_TO_PROCESS,
            single_battery_id=PROCESS_SINGLE_BATTERY_ID,
            forecast_horizon_for_target=FORECAST_HORIZON,
            add_rolling_features_flag=ADD_ROLLING_FEATURES_MAIN
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
                'internal_resistance_ohm', 'internal_resistance_ohm_lag1', 'internal_resistance_ohm_roll_mean_w5', 'internal_resistance_ohm_roll_std_w5',
                'avg_temp_C_lag1', 'avg_temp_C_roll_mean_w5',
                'is_reference_cycle', 'mission_type',
                'cycles_since_last_ref', 'SOH_at_last_ref_%'
            ]
            display_cols_present = [col for col in display_cols_subset if col in master_cycle_df.columns]
            
            if not master_cycle_df.empty:
                first_batt_id_in_master = master_cycle_df['battery_id'].iloc[0]
                print(f"\nDisplaying some columns for battery: {first_batt_id_in_master}")
                
                display_df_sample = master_cycle_df[master_cycle_df['battery_id'] == first_batt_id_in_master]
                
                skip_rows_for_display = MAX_LOOKBACK_FOR_NAN_DISPLAY
                
                start_index_for_display = skip_rows_for_display + 1 # Show rows where lag/roll features might be populated
                end_index_for_display = start_index_for_display + FORECAST_HORIZON + 8 

                if len(display_df_sample) > start_index_for_display :
                     display(display_df_sample[display_cols_present].iloc[start_index_for_display : end_index_for_display])
                else: 
                     display(display_df_sample[display_cols_present].head(10 + FORECAST_HORIZON))

            print(f"\nMaster DataFrame shape: {master_cycle_df.shape}")
            unique_batteries = master_cycle_df['battery_id'].nunique()
            print(f"Unique batteries in final DataFrame: {unique_batteries}")
            
            if target_col_name in master_cycle_df.columns:
                print(f"Number of non-NaN future SOH targets ({target_col_name}): {master_cycle_df[target_col_name].notna().sum()}")
                print(f"Number of NaN future SOH targets: {master_cycle_df[target_col_name].isna().sum()}")
            
            print("\nProcessing complete. DataFrame 'master_cycle_df' includes lag/roll features and target.")
            
            output_filename = f"processed_battery_data_h{FORECAST_HORIZON}" # Simplified name base
            if ADD_ROLLING_FEATURES_MAIN:
                output_filename += "_with_roll"
            if PROCESS_SINGLE_BATTERY_ID:
                output_filename += f"_{PROCESS_SINGLE_BATTERY_ID.replace('.', '_')}"
            else:
                output_filename += "_ALL"
            output_filename += ".csv"
            
            master_cycle_df.to_csv(output_filename, index=False)
            print(f"DataFrame saved to {output_filename}")