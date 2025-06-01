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

# Define numeric columns once for reuse
NUMERIC_COLS_RAW = ['relative_time_s', 'voltage_charger_V', 'temp_battery_C', 'voltage_load_V',
                    'current_load_A', 'temp_mosfet_C', 'temp_resistor_C']
CRITICAL_CYCLE_COLS = ['relative_time_s', 'voltage_load_V', 'current_load_A'] # Used in filter and extract

def add_sliding_window_features(df_battery, window_sizes=(3, 5, 10), feature_cols=None):
    """
    Adds sliding window features (mean, std, diff) to a DataFrame for a single battery.
    Assumes df_battery is sorted by cycle_number.
    """
    df_out = df_battery.copy()

    if feature_cols is None:
        # Define a default list of features to apply rolling windows to
        # These should be features that are expected to show trends over cycles
        feature_cols = [
            'capacity_Ah', 'energy_Wh', 'internal_resistance_ohm', 'avg_temp_C',
            'discharge_duration_s', 'SOH_cycle_capacity_%', 'avg_voltage_V',
            'delta_voltage_V', 'voltage_skewness', 'voltage_kurtosis',
            'current_skewness', 'current_kurtosis', 'temp_skewness', 'temp_kurtosis',
            'dVdQ_mean_V_mAh', 'V_slope_seg2_V_s' # Add new features here if relevant for rolling
        ]

    # Ensure 'cycle_number' exists and the DataFrame is sorted (should already be if called correctly)
    if 'cycle_number' not in df_out.columns:
        print("Warning: 'cycle_number' not in DataFrame for sliding window. Skipping.")
        return df_out
    df_out = df_out.sort_values('cycle_number')

    for col in feature_cols:
        if col not in df_out.columns:
            # print(f"Warning: Column {col} not found for sliding window. Skipping.")
            continue
        if df_out[col].isnull().all(): # Skip if column is all NaN
            # print(f"Warning: Column {col} is all NaN. Skipping for sliding window.")
            continue

        for w in window_sizes:
            # Rolling mean
            df_out[f'{col}_roll_mean_{w}'] = df_out[col].rolling(window=w, min_periods=1).mean()
            # Rolling std
            df_out[f'{col}_roll_std_{w}'] = df_out[col].rolling(window=w, min_periods=1).std()
            # Difference from W cycles ago
            df_out[f'{col}_diff_{w}'] = df_out[col].diff(periods=w)
            # Percentage change from W cycles ago (careful with zero in denominator)
            # prev_w = df_out[col].shift(w)
            # df_out[f'{col}_pct_change_{w}'] = (df_out[col] - prev_w) / prev_w.replace({0: np.nan})


    # Fill NaNs created by .diff() at the beginning, and potentially .std() if min_periods > 1
    # For .diff(), first w values will be NaN. For .std() with min_periods=1, only first is NaN if series starts with 1 value.
    # A simple backfill then fill might be too aggressive.
    # Let's leave NaNs as they are, or use a more targeted fill if needed (e.g., fill diff with 0 for first w).
    # For now, many models can handle NaNs or they can be imputed later.

    return df_out

def load_battery_data(file_path):
    """Loads and preprocesses raw battery data from a CSV file."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df.rename(columns=COLUMN_MAPPING, inplace=True, errors='ignore')
        
        for col in NUMERIC_COLS_RAW:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = np.nan # Ensure column exists if expected

        if 'mode' in df.columns:
            df['mode'] = pd.to_numeric(df['mode'], errors='coerce').fillna(99).astype(int)
        else:
            df['mode'] = 99 # Default if missing

        if 'mission_type' in df.columns:
            df['mission_type'] = pd.to_numeric(df['mission_type'], errors='coerce').fillna(0).astype(int)
        else:
            df['mission_type'] = 0 # Default if missing
            
        req_cols_raw = ['relative_time_s', 'mode', 'voltage_load_V', 'current_load_A']
        for r_col in req_cols_raw:
            if r_col not in df.columns:
                print(f"CRITICAL WARNING: Essential raw column '{r_col}' missing in {file_path}.")
                # Potentially return None or an empty DataFrame if critical columns are missing
                # For now, we'll let it proceed and downstream functions will handle missing data.
        return df
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def segment_discharge_cycles(df, battery_id):
    """Segments the raw data into individual discharge cycles using vectorized operations."""
    cycles = []
    if df is None or df.empty or 'mode' not in df.columns:
        return cycles

    # Ensure 'mode' is numeric; already done in load_battery_data, but good check
    df['mode'] = pd.to_numeric(df['mode'], errors='coerce').fillna(99).astype(int)

    # Identify discharge phases (mode == -1)
    is_discharge = df['mode'] == -1
    if not is_discharge.any(): # No discharge data
        return cycles

    # Identify start of a new discharge cycle
    # A new cycle starts when 'is_discharge' is True and the previous was False (or it's the first row)
    # .diff() will be 1 at the start of a discharge, -1 at the end
    # .cumsum() creates groups for consecutive True values
    cycle_group_indicator = (is_discharge != is_discharge.shift(1)).cumsum()
    
    # Filter for discharge periods only and assign cycle numbers
    discharge_segments = df[is_discharge]
    if discharge_segments.empty:
        return cycles
        
    # Assign a unique cycle number to each block of discharge
    # We use the cycle_group_indicator corresponding to the discharge segments
    discharge_segments = discharge_segments.assign(
        cycle_block_id=cycle_group_indicator[is_discharge]
    )

    cycle_counter = 0
    for _, group_df in discharge_segments.groupby('cycle_block_id'):
        if group_df.empty:
            continue
            
        is_ref = False
        if 'mission_type' in group_df.columns and not group_df.empty:
            # Use mode to get the most frequent mission_type, or first if all same
            # This assumes mission_type is constant within a raw discharge segment
            mt_series = pd.to_numeric(group_df['mission_type'], errors='coerce')
            if not mt_series.empty and pd.notna(mt_series.iloc[0]):
                is_ref = (mt_series.iloc[0] == 0)
        
        cycles.append({
            "battery_id": battery_id,
            "cycle_number": cycle_counter, # This will be renumbered after filtering
            "cycle_df": group_df.drop(columns=['cycle_block_id'], errors='ignore'), # Original cycle_df
            "is_reference": is_ref
        })
        cycle_counter += 1
        
    return cycles

def filter_invalid_cycles(segmented_cycles_list, battery_id):
    """Filters out invalid or short cycles based on defined criteria."""
    valid_cycles = []
    renumbered_cycle_idx = 0
    for cycle_info in segmented_cycles_list:
        df_raw_cycle = cycle_info["cycle_df"]
        # df_processing = df_raw_cycle.copy() # Avoid copy if not strictly necessary for this stage

        # Ensure critical columns exist and are numeric
        # This might be slightly redundant if load_battery_data and segment_discharge_cycles are perfect,
        # but it's a good safeguard for data integrity within this function.
        df_check = df_raw_cycle.copy() # Work on a copy for these checks/conversions
        for col in CRITICAL_CYCLE_COLS:
            if col not in df_check.columns:
                # print(f"Warning: Column {col} missing in cycle for {battery_id}, skipping cycle.")
                continue # to next cycle_info
            df_check[col] = pd.to_numeric(df_check[col], errors='coerce')
        
        df_check.dropna(subset=CRITICAL_CYCLE_COLS, inplace=True)

        if df_check.empty or len(df_check) < MINIMUM_DATA_POINTS_PER_CYCLE:
            continue

        # Use the checked and potentially cleaned df_check for voltage filtering
        first_stable_voltage_idx = df_check[df_check['voltage_load_V'] >= MIN_ACCEPTABLE_STABLE_VOLTAGE_V].index.min()
        if pd.isna(first_stable_voltage_idx):
            continue
            
        # Apply trimming to the original df_raw_cycle using the index from df_check
        df_trimmed = df_raw_cycle.loc[first_stable_voltage_idx:].copy() # Explicit copy for df_trimmed

        # Re-validate df_trimmed for critical columns and data points after trimming
        for col in CRITICAL_CYCLE_COLS: # Ensure numeric types again for the trimmed version
            if col not in df_trimmed.columns:
                 # This should not happen if CRITICAL_CYCLE_COLS were present in df_raw_cycle
                continue # to next cycle_info
            df_trimmed[col] = pd.to_numeric(df_trimmed[col], errors='coerce')
        df_trimmed.dropna(subset=CRITICAL_CYCLE_COLS, inplace=True)

        if df_trimmed.empty or len(df_trimmed) < MINIMUM_DATA_POINTS_PER_CYCLE:
            continue
        
        # Duration check (on already numeric and non-NaN 'relative_time_s')
        time_trimmed_numeric = df_trimmed['relative_time_s'] # Already numeric from above
        if len(time_trimmed_numeric) < 2: # Need at least two points for duration
            continue
        duration_trimmed = time_trimmed_numeric.iloc[-1] - time_trimmed_numeric.iloc[0]
        if duration_trimmed < MINIMUM_VALID_CYCLE_DURATION_S:
            continue

        updated_cycle_info = cycle_info.copy()
        updated_cycle_info["cycle_df"] = df_trimmed
        updated_cycle_info["cycle_number"] = renumbered_cycle_idx # Renumber valid cycles
        valid_cycles.append(updated_cycle_info)
        renumbered_cycle_idx += 1
    return valid_cycles

def extract_cycle_features(cycle_info: dict) -> dict:
    """Extracts core and statistical features from a single valid discharge cycle."""
    
    cycle_df_original = cycle_info["cycle_df"] # This is df_trimmed from filter_invalid_cycles
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
        
        # Basic statistics
        "voltage_std_V": np.nan, "voltage_variance_V2": np.nan, "voltage_skewness": np.nan,
        "voltage_kurtosis": np.nan, "voltage_p10_V": np.nan, "voltage_p25_V": np.nan,
        "voltage_p50_V": np.nan, "voltage_p75_V": np.nan, "voltage_p90_V": np.nan,
        "current_std_A": np.nan, "current_variance_A2": np.nan, "current_skewness": np.nan,
        "current_kurtosis": np.nan, "current_p10_A": np.nan, "current_p25_A": np.nan,
        "current_p50_A": np.nan, "current_p75_A": np.nan, "current_p90_A": np.nan,
        "temp_std_C": np.nan, "temp_variance_C2": np.nan, "temp_skewness": np.nan,
        "temp_kurtosis": np.nan, "temp_p10_C": np.nan, "temp_p25_C": np.nan,
        "temp_p50_C": np.nan, "temp_p75_C": np.nan, "temp_p90_C": np.nan,
        
        # Yeni DV/DQ features
        "dVdQ_mean_V_mAh": np.nan, "dVdQ_std_V_mAh": np.nan, "dVdQ_min_V_mAh": np.nan,
        "dVdQ_max_V_mAh": np.nan, "dVdQ_skewness": np.nan, "dVdQ_kurtosis": np.nan,
        
        # DENEYSEL
        # "time_to_7_0V_s": np.nan, "time_to_6_5V_s": np.nan, 
        # "time_to_6_0V_s": np.nan, "time_to_5_5V_s": np.nan,
        
        # New V-t slope segment features
        "V_slope_seg1_V_s": np.nan, "V_slope_seg2_V_s": np.nan, "V_slope_seg3_V_s": np.nan,
    }
    
    df_cleaned = cycle_df_original.copy() # Work on a copy for safety

    # Ensure 'temp_battery_C' is numeric if present
    if 'temp_battery_C' in df_cleaned.columns:
        df_cleaned['temp_battery_C'] = pd.to_numeric(df_cleaned['temp_battery_C'], errors='coerce')
    else:
        df_cleaned['temp_battery_C'] = np.nan # Add if missing for consistent structure

    # Critical columns are already numeric and non-NaN for 'relative_time_s', 'voltage_load_V', 'current_load_A'
    # due to prior processing in filter_invalid_cycles.
    # Re-check length after ensuring temp_battery_C is handled (though it's not a subset for dropna here)
    if df_cleaned.empty or len(df_cleaned) < MINIMUM_DATA_POINTS_PER_CYCLE: # Should have been caught by filter, but safety
        return features

    time_s_vals = df_cleaned['relative_time_s'].values
    current_A_vals = df_cleaned['current_load_A'].values
    voltage_V_vals = df_cleaned['voltage_load_V'].values

    if len(time_s_vals) < 2: # Need at least two points for most calculations
        return features

    features['discharge_duration_s'] = time_s_vals[-1] - time_s_vals[0]
    
    # Use np.trapz for more accurate integration if time intervals are not perfectly regular
    # Ensure current is positive for discharge capacity calculation (or use abs if definition varies)
    # For simplicity, assuming current_load_A is positive during discharge as per mode == -1.
    # If it can be negative, abs(current_A_vals) might be needed.
    delta_t_intervals = np.diff(time_s_vals)
    valid_time_mask = delta_t_intervals > 0 # Only consider positive time steps
    
    cumulative_q_mAh_cycle_steps = np.array([]) # For dVdQ

    if np.any(valid_time_mask):
        # Trapezoidal rule for capacity and energy
        avg_current_intervals = (current_A_vals[:-1][valid_time_mask] + current_A_vals[1:][valid_time_mask]) / 2.0
        features['capacity_Ah'] = np.sum(avg_current_intervals * delta_t_intervals[valid_time_mask]) / 3600.0
        
        power_W_vals = voltage_V_vals * current_A_vals # Calculate instantaneous power for all points
        avg_power_intervals = (power_W_vals[:-1][valid_time_mask] + power_W_vals[1:][valid_time_mask]) / 2.0
        features['energy_Wh'] = np.sum(avg_power_intervals * delta_t_intervals[valid_time_mask]) / 3600.0
        
        # Cumulative charge passed within this cycle for dV/dQ (mAh)
        # Using original current_A_vals and time_s_vals for full curve if possible before masking delta_t
        if len(current_A_vals) > 1 and len(time_s_vals) > 1:
            dt_for_q = np.diff(time_s_vals) # time intervals for Q
            current_midpoints_for_q = (current_A_vals[:-1] + current_A_vals[1:]) / 2.0
            q_steps_mAh = current_midpoints_for_q * dt_for_q * (1000.0 / 3600.0) # mAh for each step
            cumulative_q_mAh_cycle_steps = np.cumsum(q_steps_mAh)
            # Pad with 0 to align with voltage_V_vals for diff: V[0] corresponds to Q[0]=0
            cumulative_q_mAh_aligned = np.concatenate(([0], cumulative_q_mAh_cycle_steps))


            # dV/dQ Calculation (V/mAh)
            # Ensure cumulative_q_mAh_aligned has same length as voltage_V_vals
            # This happens if Q starts at 0 and corresponds to V[0], and Q_steps match V diffs
            if len(voltage_V_vals) == len(cumulative_q_mAh_aligned) and len(voltage_V_vals) > 1:
                dV = np.diff(voltage_V_vals)
                dQ = np.diff(cumulative_q_mAh_aligned) # dQ in mAh
                
                # Avoid division by zero or very small dQ for stability
                valid_dQ_mask_for_calc = dQ > 0.01 # Threshold for dQ in mAh, adjust if needed
                
                if np.any(valid_dQ_mask_for_calc):
                    dVdQ_curve = dV[valid_dQ_mask_for_calc] / dQ[valid_dQ_mask_for_calc] # V/mAh
                    if len(dVdQ_curve) > 0: # Ensure curve is not empty
                        features['dVdQ_mean_V_mAh'] = np.mean(dVdQ_curve)
                        features['dVdQ_std_V_mAh'] = np.std(dVdQ_curve)
                        features['dVdQ_min_V_mAh'] = np.min(dVdQ_curve)
                        features['dVdQ_max_V_mAh'] = np.max(dVdQ_curve)
                        if len(dVdQ_curve) >= 3: # Scipy stats need at least 3
                            features['dVdQ_skewness'] = scipy.stats.skew(dVdQ_curve)
                            features['dVdQ_kurtosis'] = scipy.stats.kurtosis(dVdQ_curve)
    else:
        features['capacity_Ah'] = 0.0
        features['energy_Wh'] = 0.0

    # ortalama ve başlangıç/bitim değerleri akım voltaj
    features['avg_current_A'] = np.mean(current_A_vals)
    features['avg_voltage_V'] = np.mean(voltage_V_vals)
    
    if len(voltage_V_vals)>0:
        features['start_voltage_V'] = voltage_V_vals[0]
        features['end_voltage_V'] = voltage_V_vals[-1]
        features['delta_voltage_V'] = voltage_V_vals[0] - voltage_V_vals[-1]
    
    if 'power_W_vals' in locals() and len(power_W_vals) > 0: # Check if power_W_vals was computed
        features['avg_power_W'] = np.mean(power_W_vals)
    else: # Recompute if not available
        features['avg_power_W'] = np.mean(voltage_V_vals * current_A_vals) if len(voltage_V_vals) > 0 else np.nan

    temp_C_vals = np.array([]) 
    if 'temp_battery_C' in df_cleaned.columns:
        temp_C_vals_series = df_cleaned['temp_battery_C'].dropna()
        if not temp_C_vals_series.empty:
            temp_C_vals = temp_C_vals_series.values
            if len(temp_C_vals) > 0:
                features['avg_temp_C'] = np.mean(temp_C_vals)
                features['start_temp_C'] = temp_C_vals[0]
                features['max_temp_C'] = np.max(temp_C_vals)
                if len(temp_C_vals) > 1:
                    features['end_temp_C'] = temp_C_vals[-1]
                    features['delta_temp_C'] = temp_C_vals[-1] - temp_C_vals[0]
                else: 
                    features['end_temp_C'] = temp_C_vals[0]
                    features['delta_temp_C'] = 0.0
                if len(temp_C_vals) >= 2:
                    features['temp_std_C'] = np.std(temp_C_vals)
                    features['temp_variance_C2'] = np.var(temp_C_vals)
                    percentiles_t = np.percentile(temp_C_vals, [10, 25, 50, 75, 90])
                    features['temp_p10_C'], features['temp_p25_C'], features['temp_p50_C'], \
                    features['temp_p75_C'], features['temp_p90_C'] = percentiles_t
                    if len(temp_C_vals) >= 3: 
                        features['temp_skewness'] = scipy.stats.skew(temp_C_vals)
                        features['temp_kurtosis'] = scipy.stats.kurtosis(temp_C_vals)

    # Voltage stats (require at least 2 points for std/var, 3 for skew/kurtosis)
    if len(voltage_V_vals) >= 2:
        features['voltage_std_V'] = np.std(voltage_V_vals)
        features['voltage_variance_V2'] = np.var(voltage_V_vals)
        percentiles_v = np.percentile(voltage_V_vals, [10, 25, 50, 75, 90])
        features['voltage_p10_V'], features['voltage_p25_V'], features['voltage_p50_V'], \
        features['voltage_p75_V'], features['voltage_p90_V'] = percentiles_v
        if len(voltage_V_vals) >= 3:
            features['voltage_skewness'] = scipy.stats.skew(voltage_V_vals)
            features['voltage_kurtosis'] = scipy.stats.kurtosis(voltage_V_vals)

    # Current stats (require at least 2 points for std/var, 3 for skew/kurtosis)
    if len(current_A_vals) >= 2:
        features['current_std_A'] = np.std(current_A_vals)
        features['current_variance_A2'] = np.var(current_A_vals)
        percentiles_c = np.percentile(current_A_vals, [10, 25, 50, 75, 90])
        features['current_p10_A'], features['current_p25_A'], features['current_p50_A'], \
        features['current_p75_A'], features['current_p90_A'] = percentiles_c
        if len(current_A_vals) >= 3:
            features['current_skewness'] = scipy.stats.skew(current_A_vals)
            features['current_kurtosis'] = scipy.stats.kurtosis(current_A_vals)

    # Internal Resistance Calculation
    elapsed_time_in_cleaned_cycle = time_s_vals - time_s_vals[0]
    ir_segment_mask = (elapsed_time_in_cleaned_cycle <= max(0, IR_CALCULATION_DURATION_S)) & \
                      (elapsed_time_in_cleaned_cycle >= 0)
    v_ir_segment = voltage_V_vals[ir_segment_mask]
    i_ir_segment = current_A_vals[ir_segment_mask]
    if len(v_ir_segment) > 1 and len(i_ir_segment) > 1:
        v_start_ir_calc = v_ir_segment[0]
        v_end_ir_calc = v_ir_segment[-1]
        i_avg_ir_calc_segment = np.mean(i_ir_segment) 
        if pd.notna(i_avg_ir_calc_segment) and abs(i_avg_ir_calc_segment) >= MIN_CURRENT_FOR_IR_CALC_A:
            delta_v_ir_segment = v_start_ir_calc - v_end_ir_calc 
            if delta_v_ir_segment >= 0 and i_avg_ir_calc_segment != 0: 
                features['internal_resistance_ohm'] = delta_v_ir_segment / abs(i_avg_ir_calc_segment)
    
    # New voltage slope segments            
    if len(time_s_vals) > 5 and len(voltage_V_vals) > 5: # Need enough points for meaningful regression
        cycle_duration = features['discharge_duration_s']
        if pd.notna(cycle_duration) and cycle_duration > 0:
            # Define segments based on percentage of duration
            # Correcting indices for segments
            seg1_end_idx = int(len(time_s_vals) * 0.25) # First 25% of data points
            seg2_start_idx = seg1_end_idx
            seg2_end_idx = int(len(time_s_vals) * 0.75) # Next 50% of data points
            seg3_start_idx = seg2_end_idx

            # Segment 1
            if seg1_end_idx > 1:
                t_seg1 = time_s_vals[0:seg1_end_idx] - time_s_vals[0] # Time relative to segment start
                v_seg1 = voltage_V_vals[0:seg1_end_idx]
                if len(t_seg1) > 1 and len(set(t_seg1)) > 1: # scipy.stats.linregress needs at least 2 unique x values
                    res = scipy.stats.linregress(t_seg1, v_seg1)
                    features['V_slope_seg1_V_s'] = res.slope

            # Segment 2 (middle)
            if seg2_end_idx > seg2_start_idx + 1 :
                t_seg2 = time_s_vals[seg2_start_idx:seg2_end_idx] - time_s_vals[seg2_start_idx]
                v_seg2 = voltage_V_vals[seg2_start_idx:seg2_end_idx]
                if len(t_seg2) > 1 and len(set(t_seg2)) > 1:
                    res = scipy.stats.linregress(t_seg2, v_seg2)
                    features['V_slope_seg2_V_s'] = res.slope
            
            # Segment 3 (end)
            if len(time_s_vals) > seg3_start_idx + 1:
                t_seg3 = time_s_vals[seg3_start_idx:] - time_s_vals[seg3_start_idx]
                v_seg3 = voltage_V_vals[seg3_start_idx:]
                if len(t_seg3) > 1 and len(set(t_seg3)) > 1:
                    res = scipy.stats.linregress(t_seg3, v_seg3)
                    features['V_slope_seg3_V_s'] = res.slope
                
    
    return features

def calculate_q_initial_and_soh(all_cycles_df_for_battery, nominal_capacity_ah=NOMINAL_CAPACITY_AH):
    """Calculates initial capacity (Q_initial) and State of Health (SOH) for each cycle."""
    df = all_cycles_df_for_battery.copy()
    q_initial = nominal_capacity_ah # Default

    # Prioritize reference cycles for Q_initial
    ref_q_cycles = df[
        df['is_reference_cycle'].fillna(False) & 
        df['capacity_Ah'].notna() & 
        (df['capacity_Ah'] > 0) # Ensure capacity is positive and valid
    ]
    
    if not ref_q_cycles.empty:
        # Sort by cycle_number to get the earliest reference cycle's capacity
        first_ref_cap = ref_q_cycles.sort_values('cycle_number')['capacity_Ah'].iloc[0]
        if pd.notna(first_ref_cap) and first_ref_cap > 0:
            q_initial = first_ref_cap
    else:
        # If no reference cycles, use the first valid overall cycle
        overall_q_cycles = df[
            df['capacity_Ah'].notna() & 
            (df['capacity_Ah'] > 0)
        ]
        if not overall_q_cycles.empty:
            first_overall_cap = overall_q_cycles.sort_values('cycle_number')['capacity_Ah'].iloc[0]
            if pd.notna(first_overall_cap) and first_overall_cap > 0:
                q_initial = first_overall_cap
                
    df['q_initial_Ah'] = q_initial
    
    if q_initial > 0 and 'capacity_Ah' in df.columns:
        df['SOH_cycle_capacity_%'] = (df['capacity_Ah'] / q_initial) * 100.0
    else:
        df['SOH_cycle_capacity_%'] = np.nan
        
    return df

def add_health_indicators(all_cycles_df_for_battery):
    """Adds various health indicator features based on cycle data. (Currently Minimal)"""
    df = all_cycles_df_for_battery.copy()
    # The specific HIs were removed.
    return df

def process_battery_dataset(root_dir, folders_list, single_battery_id=None):
    """Orchestrator to process dataset: load, segment, filter, extract features, add SOH."""
    all_batteries_processed_list = []
    found_single_battery_processed = False

    for folder_name in folders_list:
        if found_single_battery_processed and single_battery_id:
            break
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder {folder_path} not found.")
            continue
        
        print(f"Scanning folder: {folder_path}")
        # sorted listdir for consistent processing order if needed for debugging/reproducibility
        for file_name in sorted(os.listdir(folder_path)): 
            if file_name.endswith(".csv"):
                current_batt_id = os.path.splitext(file_name)[0]

                if single_battery_id and current_batt_id != single_battery_id:
                    continue
                
                # Print only if it's the target single battery or if processing all
                if not single_battery_id or (single_battery_id and not found_single_battery_processed):
                    print(f"\nProcessing battery: {current_batt_id} from file {file_name}")
                
                if single_battery_id and current_batt_id == single_battery_id:
                    found_single_battery_processed = True

                file_full_path = os.path.join(folder_path, file_name)
                raw_df = load_battery_data(file_full_path)

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
                    # Only add if capacity is valid, as it's a key metric
                    if 'capacity_Ah' in cycle_features and pd.notna(cycle_features['capacity_Ah']) and cycle_features['capacity_Ah'] > 0.01: 
                         battery_features_list.append(cycle_features)
                
                if not battery_features_list:
                    print(f"  No features extracted from valid cycles for {current_batt_id}.")
                    if single_battery_id and current_batt_id == single_battery_id: break
                    continue
                    
                df_one_batt = pd.DataFrame(battery_features_list)
                df_one_batt = df_one_batt.sort_values('cycle_number').reset_index(drop=True)
                
                df_with_soh = calculate_q_initial_and_soh(df_one_batt, NOMINAL_CAPACITY_AH)
                df_enriched = add_health_indicators(df_with_soh)
                
                if not df_enriched.empty:
                    # print(f"  Adding sliding window features for {current_batt_id}...")
                    df_enriched = add_sliding_window_features(df_enriched)
                
                all_batteries_processed_list.append(df_enriched)

                if single_battery_id and current_batt_id == single_battery_id:
                    print(f"  Finished processing single battery: {current_batt_id}")
                    break # Break from files loop
        
        if single_battery_id and found_single_battery_processed:
            break # Break from folders loop

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
        PROCESS_SINGLE_BATTERY_ID = None # Example: "b1c0" or None for all
        # PROCESS_SINGLE_BATTERY_ID = "regular_battery_001" # For testing a single battery
        
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
            display_cols_subset = [
                'battery_id', 'cycle_number', 'is_reference_cycle',
                'SOH_cycle_capacity_%', 'capacity_Ah', 'internal_resistance_ohm',
                'avg_temp_C', 'voltage_skewness', 'current_kurtosis', 'temp_p50_C'
            ]
            display_cols_present = [col for col in display_cols_subset if col in master_cycle_df.columns]
            
            if not master_cycle_df.empty and display_cols_present:
                # Display sample data for the first battery_id found in the master_df
                # This ensures it works even if a single_battery_id was processed that isn't the absolute first alphabetically
                first_batt_id_in_master = master_cycle_df['battery_id'].iloc[0]
                print(f"\nSample data for battery: {first_batt_id_in_master} (some columns)")
                display(master_cycle_df[master_cycle_df['battery_id'] == first_batt_id_in_master][display_cols_present].head(15))
            
            print(f"\nMaster DataFrame final shape: {master_cycle_df.shape}")
            print(f"Unique batteries: {master_cycle_df['battery_id'].nunique()}")
            print("\n--- Processing complete (Further Simplified Features). ---")
            
            output_filename_base = "processed_battery_data_further_simplified"
            output_filename_suffix = ""
            if PROCESS_SINGLE_BATTERY_ID:
                sanitized_id = PROCESS_SINGLE_BATTERY_ID.replace('.', '_').replace('/', '_')
                output_filename_suffix = f"_{sanitized_id}"
            else:
                output_filename_suffix = "_ALL"
            
            output_filename = f"{output_filename_base}{output_filename_suffix}.csv"
            
            try:
                master_cycle_df.to_csv(output_filename, index=False)
                print(f"Final DataFrame saved to: {output_filename}")
            except Exception as e:
                print(f"Error saving final DataFrame: {e}")

# --- END OF FILE process_data.py ---