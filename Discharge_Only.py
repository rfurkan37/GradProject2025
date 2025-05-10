import pandas as pd
import os
from dataclasses import dataclass
from typing import List, Tuple
import traceback


@dataclass
class DatasetFile:
    source_path: str
    destination_directory: str


def extract_discharge_segments(dataset: DatasetFile, min_segment_rows=10, min_voltage_drop=0.5, max_negative_percent=20):
    """
    Process a single battery CSV file and extract only valid discharge segments to separate files.
    
    Args:
        dataset: DatasetFile containing source path and destination directory
        min_segment_rows: Minimum number of rows for a valid discharge segment
        min_voltage_drop: Minimum voltage drop required (V) for a valid discharge segment
        max_negative_percent: Maximum allowed percentage of negative voltage values (0-100)
    """
    file_path = dataset.source_path
    output_dir = dataset.destination_directory
    
    print(f"Processing {file_path}...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Find all continuous discharge segments
    discharge_segments = []
    start_idx = None
    
    # Identify discharge segments (where mode == -1)
    for i in range(len(df)):
        if df.iloc[i]['mode'] == -1:  # Discharge mode
            if start_idx is None:  # Start of a new discharge segment
                start_idx = i
        elif start_idx is not None:  # End of a discharge segment
            discharge_segments.append((start_idx, i))
            start_idx = None
    
    # Don't forget the last segment if it ends at EOF
    if start_idx is not None:
        discharge_segments.append((start_idx, len(df)))
    
    print(f"Found {len(discharge_segments)} potential discharge segments")
    
    valid_segments = 0
    skipped_segments = 0
    
    # Counter for consecutive file naming
    output_file_counter = 1
    
    # Process each discharge segment
    for segment_idx, (start_idx, end_idx) in enumerate(discharge_segments, 1):
        # Extract segment data
        segment_data = df.iloc[start_idx:end_idx].copy()
        
        # Verify this is actually a discharge segment
        if segment_data['mode'].iloc[0] != -1:
            print(f"Warning: Original segment {segment_idx} does not start with discharge mode. Skipping.")
            skipped_segments += 1
            continue
        
        # VALIDATION 1: Check segment length
        if len(segment_data) < min_segment_rows:
            print(f"Warning: Original segment {segment_idx} has only {len(segment_data)} rows (min {min_segment_rows}). Skipping.")
            skipped_segments += 1
            continue
        
        # VALIDATION 2: Check percentage of negative voltage values
        negative_values = (segment_data['voltage_load'] < 0).sum()
        negative_percent = (negative_values / len(segment_data)) * 100
        
        if negative_percent > max_negative_percent:
            print(f"Warning: Original segment {segment_idx} has {negative_percent:.1f}% negative voltage values (max {max_negative_percent}%). Skipping.")
            skipped_segments += 1
            continue
        else:
            # Filter out rows with negative voltage values
            if negative_values > 0:
                print(f"Info: Original segment {segment_idx} has {negative_values} negative voltage rows ({negative_percent:.1f}%). Filtering these rows.")
                segment_data = segment_data[segment_data['voltage_load'] >= 0].reset_index(drop=True)
        
        # If after filtering, the segment is too small, skip it
        if len(segment_data) < min_segment_rows:
            print(f"Warning: Original segment {segment_idx} has only {len(segment_data)} rows after filtering (min {min_segment_rows}). Skipping.")
            skipped_segments += 1
            continue
            
        # VALIDATION 3: Check voltage pattern (should decrease)
        # Get first and last non-zero voltage readings
        valid_voltage_rows = segment_data[segment_data['voltage_load'] > 0]
        if len(valid_voltage_rows) < 2:
            print(f"Warning: Original segment {segment_idx} doesn't have enough valid voltage readings. Skipping.")
            skipped_segments += 1
            continue
            
        start_voltage = valid_voltage_rows['voltage_load'].iloc[0]
        end_voltage = valid_voltage_rows['voltage_load'].iloc[-1]
        
        if start_voltage <= end_voltage:
            print(f"Warning: Original segment {segment_idx} doesn't show expected voltage drop pattern. Skipping.")
            skipped_segments += 1
            continue
            
        # VALIDATION 4: Check for minimum voltage drop
        voltage_drop = start_voltage - end_voltage
        if voltage_drop < min_voltage_drop:
            print(f"Warning: Original segment {segment_idx} has insufficient voltage drop ({voltage_drop:.2f}V). Skipping.")
            skipped_segments += 1
            continue
            
        # Check if this is a reference discharge
        is_reference = False
        if 'mission_type' in segment_data.columns:
            # If any row in this discharge segment has mission_type == 0, it's a reference discharge
            is_reference = (segment_data['mission_type'] == 0).any()
        
        # Create filename with reference indication if applicable and consecutive numbering
        if is_reference:
            output_path = os.path.join(output_dir, f"discharge{output_file_counter}_reference.csv")
        else:
            output_path = os.path.join(output_dir, f"discharge{output_file_counter}.csv")
        
        # Save to CSV
        segment_data.to_csv(output_path, index=False)
        
        discharge_type = "reference" if is_reference else "regular"
        print(f"Discharge segment {output_file_counter} ({discharge_type}) written to {output_path} ({len(segment_data)} rows) [original segment {segment_idx}]")
        
        valid_segments += 1
        output_file_counter += 1  # Increment counter for next file
    
    print(f"Processed {len(discharge_segments)} segments: {valid_segments} valid, {skipped_segments} skipped")
    return valid_segments, skipped_segments


def main():
    # Define validation parameters
    MIN_ROWS = 10           # Minimum number of rows for a valid discharge
    MIN_VOLTAGE_DROP = 0.5  # Minimum voltage drop in volts
    MAX_NEGATIVE_PERCENT = 20  # Maximum percentage of negative values allowed (0-100)
    
    datasets = [
        # Regular Alt Batteries
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery00.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery00"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery01.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery01"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery10.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery10"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery11.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery11"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery20.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery20"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery21.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery21"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery22.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery22"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery23.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery23"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery30.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery30"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery31.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery31"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery40.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery40"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery41.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery41"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery50.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery50"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery51.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery51"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery52.csv", destination_directory= "../../Discharge_Only_Dataset/regular_alt_batteries/battery52"),

        # Recommissioned Batteries
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery02.csv", destination_directory= "../../Discharge_Only_Dataset/recommissioned_batteries/battery02"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery03.csv", destination_directory= "../../Discharge_Only_Dataset/recommissioned_batteries/battery03"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery12.csv", destination_directory= "../../Discharge_Only_Dataset/recommissioned_batteries/battery12"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery24.csv", destination_directory= "../../Discharge_Only_Dataset/recommissioned_batteries/battery24"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery25.csv", destination_directory= "../../Discharge_Only_Dataset/recommissioned_batteries/battery25"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery32.csv", destination_directory= "../../Discharge_Only_Dataset/recommissioned_batteries/battery32"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery33.csv", destination_directory= "../../Discharge_Only_Dataset/recommissioned_batteries/battery33"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery53.csv", destination_directory= "../../Discharge_Only_Dataset/recommissioned_batteries/battery53"),

        # Second Life Batteries
        DatasetFile(source_path= "../../Datasets/second_life_batteries/battery13.csv", destination_directory= "../../Discharge_Only_Dataset/second_life_batteries/battery13"),
        DatasetFile(source_path= "../../Datasets/second_life_batteries/battery36.csv", destination_directory= "../../Discharge_Only_Dataset/second_life_batteries/battery36"),
        DatasetFile(source_path= "../../Datasets/second_life_batteries/battery54.csv", destination_directory= "../../Discharge_Only_Dataset/second_life_batteries/battery54"),
    ]
    
    total_valid = 0
    total_skipped = 0
    
    # Process each dataset
    for dataset in datasets:
        if os.path.exists(dataset.source_path):
            try:
                valid, skipped = extract_discharge_segments(
                    dataset, 
                    min_segment_rows=MIN_ROWS, 
                    min_voltage_drop=MIN_VOLTAGE_DROP,
                    max_negative_percent=MAX_NEGATIVE_PERCENT
                )
                total_valid += valid
                total_skipped += skipped
                print(f"Successfully processed {dataset.source_path}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing {dataset.source_path}: {str(e)}")
                traceback.print_exc()  # Print full traceback
        else:
            print(f"Warning: File {dataset.source_path} does not exist. Skipping...")
    
    print(f"Total segments processed: {total_valid + total_skipped}")
    print(f"Total valid segments: {total_valid}")
    print(f"Total skipped segments: {total_skipped}")


if __name__ == "__main__":
    main()