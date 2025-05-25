# --- START OF FILE process_data.py ---

import pandas as pd
import numpy as np
import os
from IPython.display import display # For better display in notebooks
import scipy.stats # For skewness, kurtosis

# --- Configuration Constants ---
DATASET_ROOT_DIRECTORY = "battery_alt_dataset"
REGULAR_BATTERIES_FOLDER = "regular_alt_batteries"
RECOMMISSIONED_BATTERIES_FOLDER = "recommissioned_batteries"
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
MIN_CURRENT_FOR_IR_CALC_A = 0.5

def load_battery_data(file_path):
    """Loads and preprocesses raw battery data from a CSV file."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df.rename(columns=COLUMN_MAPPING, inplace=True, errors='ignore')
        num_cols = ['relative_time_s', 'voltage_charger_V', 'temp_battery_C', 'voltage_load_V',
                    'current_load_A', 'temp_mosfet_C', 'temp_resistor_C']
        for col in num_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            else: df[col] = np.nan
        if 'mode' in df.columns: df['mode'] = pd.to_numeric(df['mode'], errors='coerce').fillna(99).astype(int)
        else: df['mode'] = 99
        if 'mission_type' in df.columns: df['mission_type'] = pd.to_numeric(df['mission_type'], errors='coerce').fillna(0).astype(int)
        else: df['mission_type'] = 0
        req_cols_raw = ['relative_time_s', 'mode', 'voltage_load_V', 'current_load_A']
        for r_col in req_cols_raw:
            if r_col not in df.columns: print(f"CRITICAL WARNING: Essential raw column '{r_col}' missing in {file_path}.")
        return df
    except FileNotFoundError: print(f"Error: File not found {file_path}"); return None
    except Exception as e: print(f"Error loading {file_path}: {e}"); return None

def segment_discharge_cycles(df, battery_id):
    """Segments the raw data into individual discharge cycles."""
    cycles = []
    if df is None or df.empty or 'mode' not in df.columns: return cycles
    df['mode'] = pd.to_numeric(df['mode'], errors='coerce').fillna(99).astype(int)
    in_discharge_phase = False
    current_cycle_rows = []
    cycle_counter = 0
    for index, row in df.iterrows():
        if row['mode'] == -1:
            if not in_discharge_phase: in_discharge_phase = True; current_cycle_rows = []
            current_cycle_rows.append(row.copy())
        elif in_discharge_phase:
            in_discharge_phase = False
            if current_cycle_rows:
                cycle_df = pd.DataFrame(current_cycle_rows); is_ref = False
                if 'mission_type' in cycle_df.columns and not cycle_df.empty:
                    mt = pd.to_numeric(cycle_df['mission_type'].iloc[0], errors='coerce')
                    if pd.notna(mt): is_ref = (mt == 0)
                cycles.append({"battery_id": battery_id, "cycle_number": cycle_counter, "cycle_df": cycle_df, "is_reference": is_ref})
                cycle_counter += 1; current_cycle_rows = []
    if in_discharge_phase and current_cycle_rows:
        cycle_df = pd.DataFrame(current_cycle_rows); is_ref = False
        if 'mission_type' in cycle_df.columns and not cycle_df.empty:
            mt = pd.to_numeric(cycle_df['mission_type'].iloc[0], errors='coerce')
            if pd.notna(mt): is_ref = (mt == 0)
        cycles.append({"battery_id": battery_id, "cycle_number": cycle_counter, "cycle_df": cycle_df, "is_reference": is_ref})
    return cycles

def filter_invalid_cycles(segmented_cycles_list, battery_id):
    """Filters out invalid or short cycles based on defined criteria."""
    valid_cycles = []
    renumbered_cycle_idx = 0
    for cycle_info in segmented_cycles_list:
        df_raw_cycle = cycle_info["cycle_df"]; df_processing = df_raw_cycle.copy()
        crit_cols = ['relative_time_s', 'voltage_load_V', 'current_load_A']
        if any(col not in df_processing.columns for col in crit_cols): continue
        for col in crit_cols: df_processing[col] = pd.to_numeric(df_processing[col], errors='coerce')
        df_processing.dropna(subset=crit_cols, inplace=True)
        if df_processing.empty or len(df_processing) < MINIMUM_DATA_POINTS_PER_CYCLE: continue
        first_stable_voltage_idx = df_processing[df_processing['voltage_load_V'] >= MIN_ACCEPTABLE_STABLE_VOLTAGE_V].index.min()
        if pd.isna(first_stable_voltage_idx): continue
        df_trimmed = df_raw_cycle.loc[first_stable_voltage_idx:].copy()
        for col in crit_cols: df_trimmed[col] = pd.to_numeric(df_trimmed[col], errors='coerce')
        df_trimmed.dropna(subset=crit_cols, inplace=True)
        if df_trimmed.empty or len(df_trimmed) < MINIMUM_DATA_POINTS_PER_CYCLE: continue
        time_trimmed_numeric = pd.to_numeric(df_trimmed['relative_time_s'], errors='coerce').dropna()
        if len(time_trimmed_numeric) < 2: continue
        duration_trimmed = time_trimmed_numeric.iloc[-1] - time_trimmed_numeric.iloc[0]
        if duration_trimmed < MINIMUM_VALID_CYCLE_DURATION_S: continue
        updated_cycle_info = cycle_info.copy()
        updated_cycle_info["cycle_df"] = df_trimmed
        updated_cycle_info["cycle_number"] = renumbered_cycle_idx
        valid_cycles.append(updated_cycle_info)
        renumbered_cycle_idx += 1
    return valid_cycles

def extract_cycle_features(cycle_info: dict) -> dict:
    """Extracts core and statistical features from a single valid discharge cycle."""
    cycle_df_original = cycle_info["cycle_df"]
    features = {
        "battery_id": cycle_info["battery_id"],
        "cycle_number": cycle_info["cycle_number"],
        "is_reference_cycle": cycle_info.get("is_reference", False),
        "discharge_duration_s": np.nan, "capacity_Ah": np.nan, "energy_Wh": np.nan,
        "avg_current_A": np.nan, "avg_voltage_V": np.nan,
        "start_voltage_V": np.nan, "end_voltage_V": np.nan, "delta_voltage_V": np.nan,
        "avg_power_W": np.nan, "avg_temp_C": np.nan, "start_temp_C": np.nan,
        "end_temp_C": np.nan, "delta_temp_C": np.nan, "max_temp_C": np.nan,
        "internal_resistance_ohm": np.nan,
        "voltage_std_V": np.nan, "voltage_variance_V2": np.nan, "voltage_skewness": np.nan,
        "voltage_kurtosis": np.nan, "voltage_p10_V": np.nan, "voltage_p25_V": np.nan,
        "voltage_p50_V": np.nan, "voltage_p75_V": np.nan, "voltage_p90_V": np.nan,
        "current_std_A": np.nan, "current_variance_A2": np.nan, "current_skewness": np.nan,
        "current_kurtosis": np.nan, "current_p10_A": np.nan, "current_p25_A": np.nan,
        "current_p50_A": np.nan, "current_p75_A": np.nan, "current_p90_A": np.nan,
        "temp_std_C": np.nan, "temp_variance_C2": np.nan, "temp_skewness": np.nan,
        "temp_kurtosis": np.nan, "temp_p10_C": np.nan, "temp_p25_C": np.nan,
        "temp_p50_C": np.nan, "temp_p75_C": np.nan, "temp_p90_C": np.nan,
    }
    
    df_cleaned = cycle_df_original.copy()
    std_cols_for_calc = ['relative_time_s', 'voltage_load_V', 'current_load_A', 'temp_battery_C']
    for col in std_cols_for_calc:
        if col in df_cleaned.columns: df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        else: df_cleaned[col] = np.nan
    df_cleaned.dropna(subset=['relative_time_s', 'voltage_load_V', 'current_load_A'], inplace=True)
    if df_cleaned.empty or len(df_cleaned) < MINIMUM_DATA_POINTS_PER_CYCLE: return features

    time_s_vals = df_cleaned['relative_time_s'].values
    current_A_vals = df_cleaned['current_load_A'].values
    voltage_V_vals = df_cleaned['voltage_load_V'].values

    features['discharge_duration_s'] = time_s_vals[-1] - time_s_vals[0]
    if len(time_s_vals) > 1:
        delta_t_intervals = np.diff(time_s_vals); valid_time_mask = delta_t_intervals > 0
        if np.any(valid_time_mask):
            avg_current_intervals = (current_A_vals[:-1][valid_time_mask] + current_A_vals[1:][valid_time_mask]) / 2.0
            features['capacity_Ah'] = np.sum(avg_current_intervals * delta_t_intervals[valid_time_mask]) / 3600.0
            power_W_vals = voltage_V_vals * current_A_vals
            avg_power_intervals = (power_W_vals[:-1][valid_time_mask] + power_W_vals[1:][valid_time_mask]) / 2.0
            features['energy_Wh'] = np.sum(avg_power_intervals * delta_t_intervals[valid_time_mask]) / 3600.0
        else: features['capacity_Ah'] = 0.0; features['energy_Wh'] = 0.0
    else: features['capacity_Ah'] = 0.0; features['energy_Wh'] = 0.0

    features['avg_current_A'] = np.mean(current_A_vals)
    features['avg_voltage_V'] = np.mean(voltage_V_vals)
    features['start_voltage_V'] = voltage_V_vals[0]
    features['end_voltage_V'] = voltage_V_vals[-1]
    features['delta_voltage_V'] = voltage_V_vals[0] - voltage_V_vals[-1]
    features['avg_power_W'] = np.mean(voltage_V_vals * current_A_vals)

    temp_C_vals = np.array([])
    if 'temp_battery_C' in df_cleaned.columns and df_cleaned['temp_battery_C'].notna().any():
        temp_C_vals_series = df_cleaned['temp_battery_C'].dropna()
        if not temp_C_vals_series.empty:
            temp_C_vals = temp_C_vals_series.values
            features['avg_temp_C'] = np.mean(temp_C_vals); features['start_temp_C'] = temp_C_vals[0]
            features['end_temp_C'] = temp_C_vals[-1]; features['max_temp_C'] = np.max(temp_C_vals)
            features['delta_temp_C'] = (temp_C_vals[-1] - temp_C_vals[0]) if len(temp_C_vals) > 1 else 0.0
            if len(temp_C_vals) >= 2:
                features['temp_std_C'] = np.std(temp_C_vals); features['temp_variance_C2'] = np.var(temp_C_vals)
                if len(temp_C_vals) >= 3:
                    features['temp_skewness'] = scipy.stats.skew(temp_C_vals)
                    features['temp_kurtosis'] = scipy.stats.kurtosis(temp_C_vals)
                percentiles_t = np.percentile(temp_C_vals, [10, 25, 50, 75, 90])
                features['temp_p10_C'], features['temp_p25_C'], features['temp_p50_C'], \
                features['temp_p75_C'], features['temp_p90_C'] = percentiles_t

    if len(voltage_V_vals) >= 2:
        features['voltage_std_V'] = np.std(voltage_V_vals); features['voltage_variance_V2'] = np.var(voltage_V_vals)
        if len(voltage_V_vals) >= 3:
            features['voltage_skewness'] = scipy.stats.skew(voltage_V_vals)
            features['voltage_kurtosis'] = scipy.stats.kurtosis(voltage_V_vals)
        percentiles_v = np.percentile(voltage_V_vals, [10, 25, 50, 75, 90])
        features['voltage_p10_V'], features['voltage_p25_V'], features['voltage_p50_V'], \
        features['voltage_p75_V'], features['voltage_p90_V'] = percentiles_v

    if len(current_A_vals) >= 2:
        features['current_std_A'] = np.std(current_A_vals); features['current_variance_A2'] = np.var(current_A_vals)
        if len(current_A_vals) >= 3:
            features['current_skewness'] = scipy.stats.skew(current_A_vals)
            features['current_kurtosis'] = scipy.stats.kurtosis(current_A_vals)
        percentiles_c = np.percentile(current_A_vals, [10, 25, 50, 75, 90])
        features['current_p10_A'], features['current_p25_A'], features['current_p50_A'], \
        features['current_p75_A'], features['current_p90_A'] = percentiles_c

    if len(time_s_vals) > 1:
        elapsed_time_in_cleaned_cycle = time_s_vals - time_s_vals[0]
        ir_segment_mask = (elapsed_time_in_cleaned_cycle <= max(0, IR_CALCULATION_DURATION_S)) & (elapsed_time_in_cleaned_cycle >= 0)
        v_ir_segment = voltage_V_vals[ir_segment_mask]; i_ir_segment = current_A_vals[ir_segment_mask]
        if len(v_ir_segment) > 1 and len(i_ir_segment) > 1:
            v_start_ir_calc = v_ir_segment[0]; v_end_ir_calc = v_ir_segment[-1]
            i_avg_ir_calc_segment = np.mean(i_ir_segment)
            if pd.notna(i_avg_ir_calc_segment) and abs(i_avg_ir_calc_segment) >= MIN_CURRENT_FOR_IR_CALC_A:
                delta_v_ir_segment = v_start_ir_calc - v_end_ir_calc
                if delta_v_ir_segment >= 0 and i_avg_ir_calc_segment != 0:
                    features['internal_resistance_ohm'] = delta_v_ir_segment / abs(i_avg_ir_calc_segment)
    return features

def calculate_q_initial_and_soh(all_cycles_df_for_battery, nominal_capacity_ah=NOMINAL_CAPACITY_AH):
    """Calculates initial capacity (Q_initial) and State of Health (SOH) for each cycle."""
    df = all_cycles_df_for_battery.copy(); q_initial = nominal_capacity_ah
    ref_q_cycles = df[df['is_reference_cycle'].fillna(False) & df['capacity_Ah'].notna() & (df['capacity_Ah'] > 0)]
    if not ref_q_cycles.empty:
        first_ref_cap = ref_q_cycles.sort_values('cycle_number')['capacity_Ah'].iloc[0]
        if pd.notna(first_ref_cap) and first_ref_cap > 0: q_initial = first_ref_cap
    else:
        overall_q_cycles = df[df['capacity_Ah'].notna() & (df['capacity_Ah'] > 0)]
        if not overall_q_cycles.empty:
            first_overall_cap = overall_q_cycles.sort_values('cycle_number')['capacity_Ah'].iloc[0]
            if pd.notna(first_overall_cap) and first_overall_cap > 0: q_initial = first_overall_cap
    df['q_initial_Ah'] = q_initial
    if q_initial > 0 and 'capacity_Ah' in df.columns: df['SOH_cycle_capacity_%'] = (df['capacity_Ah'] / q_initial) * 100.0
    else: df['SOH_cycle_capacity_%'] = np.nan
    return df

def add_health_indicators(all_cycles_df_for_battery):
    """Adds various health indicator features based on cycle data. (Currently Minimal)"""
    df = all_cycles_df_for_battery.copy()
    # The specific HIs (HI_voltage_ratio_end_start, HI_capacity_fade_from_prev_cycle_Ah,
    # HI_IR_change_from_prev_cycle_ohm) have been removed as per request.
    # If you add other HIs in the future, they would go here.
    # For now, this function might not add any new columns if all are removed.
    return df

def process_battery_dataset(root_dir, folders_list, single_battery_id=None):
    """Orchestrator to process dataset: load, segment, filter, extract features, add SOH."""
    all_batteries_processed_list = []
    found_single_battery_processed = False
    for folder_name in folders_list:
        if found_single_battery_processed and single_battery_id: break
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path): print(f"Warning: Folder {folder_path} not found."); continue
        print(f"Scanning folder: {folder_path}")
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith(".csv"):
                current_batt_id = os.path.splitext(file_name)[0]
                if single_battery_id and current_batt_id != single_battery_id: continue
                if not single_battery_id or (single_battery_id and not found_single_battery_processed):
                    print(f"\nProcessing battery: {current_batt_id} from file {file_name}")
                if single_battery_id and current_batt_id == single_battery_id: found_single_battery_processed = True

                raw_df = load_battery_data(os.path.join(folder_path, file_name))
                if raw_df is None or raw_df.empty:
                    print(f"  No data loaded for {current_batt_id}.")
                    if single_battery_id and current_batt_id == single_battery_id: break
                    continue
                segmented_cycles = segment_discharge_cycles(raw_df, current_batt_id)
                if not segmented_cycles:
                    print(f"  No discharge cycles segmented for {current_batt_id}.")
                    if single_battery_id and current_batt_id == single_battery_id: break
                    continue
                valid_cycles_info_list = filter_invalid_cycles(segmented_cycles, current_batt_id)
                if not valid_cycles_info_list:
                    print(f"  No valid cycles after filtering for {current_batt_id}.")
                    if single_battery_id and current_batt_id == single_battery_id: break
                    continue
                battery_features_list = []
                for cycle_data_info in valid_cycles_info_list:
                    cycle_features = extract_cycle_features(cycle_data_info)
                    if 'capacity_Ah' in cycle_features and pd.notna(cycle_features['capacity_Ah']):
                         battery_features_list.append(cycle_features)
                if not battery_features_list:
                    print(f"  No features extracted from valid cycles for {current_batt_id}.")
                    if single_battery_id and current_batt_id == single_battery_id: break
                    continue
                df_one_batt = pd.DataFrame(battery_features_list)
                df_one_batt = df_one_batt.sort_values('cycle_number').reset_index(drop=True)
                df_with_soh = calculate_q_initial_and_soh(df_one_batt, NOMINAL_CAPACITY_AH)
                df_enriched = add_health_indicators(df_with_soh) # This function now does less/nothing
                all_batteries_processed_list.append(df_enriched)
                if single_battery_id and current_batt_id == single_battery_id:
                    print(f"  Finished processing single battery: {current_batt_id}"); break
        if single_battery_id and found_single_battery_processed: break

    if not all_batteries_processed_list:
        print("No battery data successfully processed. Master DataFrame will be empty.")
        return pd.DataFrame()
    print("\nConcatenating data from all processed batteries...")
    master_df = pd.concat(all_batteries_processed_list, ignore_index=True)
    master_df = master_df.sort_values(['battery_id', 'cycle_number']).reset_index(drop=True)
    print("Feature extraction and SOH calculation complete.")
    return master_df

# --- Main Execution Block ---
if __name__ == "__main__":
    if not os.path.exists(DATASET_ROOT_DIRECTORY):
         print(f"!!! CRITICAL: DATASET_ROOT_DIRECTORY '{DATASET_ROOT_DIRECTORY}' not found. !!!")
    else:
        PROCESS_SINGLE_BATTERY_ID = None
        print(f"--- Script Mode: {'SINGLE battery: ' + PROCESS_SINGLE_BATTERY_ID if PROCESS_SINGLE_BATTERY_ID else 'ALL batteries'} ---")
        master_cycle_df = process_battery_dataset(
            DATASET_ROOT_DIRECTORY,
            FOLDERS_TO_PROCESS,
            single_battery_id=PROCESS_SINGLE_BATTERY_ID
        )
        if master_cycle_df.empty:
            print("\n--- Processing Result: No data processed. ---")
        else:
            print("\n\n--- Master DataFrame Summary (Further Simplified Features) ---")
            # Note: 'mission_type' is not in the features dict, so it won't be in master_cycle_df
            # 'is_reference_cycle' is still present.
            display_cols_subset = [
                'battery_id', 'cycle_number', 'is_reference_cycle',
                'SOH_cycle_capacity_%', 'capacity_Ah', 'internal_resistance_ohm',
                'avg_temp_C', 'voltage_skewness', 'current_kurtosis', 'temp_p50_C'
            ]
            display_cols_present = [col for col in display_cols_subset if col in master_cycle_df.columns]
            if not master_cycle_df.empty and display_cols_present:
                first_batt_id = master_cycle_df['battery_id'].iloc[0]
                print(f"\nSample data for battery: {first_batt_id} (some columns)")
                display(master_cycle_df[master_cycle_df['battery_id'] == first_batt_id][display_cols_present].head(15))
            print(f"\nMaster DataFrame final shape: {master_cycle_df.shape}")
            print(f"Unique batteries: {master_cycle_df['battery_id'].nunique()}")
            print("\n--- Processing complete (Further Simplified Features). ---")
            output_filename = f"processed_battery_data_further_simplified"
            if PROCESS_SINGLE_BATTERY_ID:
                sanitized_id = PROCESS_SINGLE_BATTERY_ID.replace('.', '_').replace('/', '_')
                output_filename += f"_{sanitized_id}"
            else: output_filename += "_ALL"
            output_filename += ".csv"
            try:
                master_cycle_df.to_csv(output_filename, index=False)
                print(f"Final DataFrame saved to: {output_filename}")
            except Exception as e: print(f"Error saving final DataFrame: {e}")
# --- END OF FILE process_data.py ---