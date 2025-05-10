import pandas as pd
import os
from dataclasses import dataclass
from typing import List


@dataclass
class DatasetFile:
    source_path: str
    destination_directory: str


def process_battery_file(dataset: DatasetFile):
    """
    Process a single battery CSV file and extract cycles to separate files.
    Each cycle is defined as:
    steady(pre-discharge) → discharge → steady(post-discharge) → charge → steady(post-charge)
    
    Args:
        dataset: DatasetFile containing source path and destination directory
    """
    file_path = dataset.source_path
    output_dir = dataset.destination_directory
    
    print(f"Processing {file_path}...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Map the numeric mode to stage names for better readability
    mode_to_stage = {-1: 'discharge', 1: 'charge', 0: 'steady'}
    
    # Add a 'stage' column
    df['stage'] = df['mode'].map(mode_to_stage)
    
    # Find all transitions from steady to discharge (start of a new cycle)
    cycle_start_indices = []
    
    # Look for transitions from steady to discharge
    for i in range(1, len(df)):
        if df.iloc[i]['mode'] == -1 and df.iloc[i-1]['mode'] == 0:
            cycle_start_indices.append(i-1)  # Include the pre-discharge steady state
    
    print(f"Found {len(cycle_start_indices)} potential cycle starts")
    
    if len(cycle_start_indices) == 0:
        print("No complete cycles found. Exiting.")
        return
    
    # Process each cycle
    valid_cycles = 0
    
    for cycle_idx in range(len(cycle_start_indices)):
        # Determine start and end indices for this cycle
        start_idx = cycle_start_indices[cycle_idx]
        end_idx = cycle_start_indices[cycle_idx + 1] if cycle_idx + 1 < len(cycle_start_indices) else len(df)
        
        # Extract cycle data
        cycle_data = df.iloc[start_idx:end_idx].copy()
        
        # Verify this is a complete cycle by checking the sequence of modes
        # We need: steady → discharge → steady → charge → steady
        expected_sequence = [0, -1, 0, 1, 0]
        
        # Check if the cycle contains all expected modes in order
        has_complete_sequence = True
        current_mode_idx = 0
        
        for _, row in cycle_data.iterrows():
            mode = row['mode']
            
            # If we're looking for the current mode in the sequence
            if mode == expected_sequence[current_mode_idx]:
                # Move to the next expected mode
                if current_mode_idx < len(expected_sequence) - 1:
                    current_mode_idx += 1
            # If we see the same mode again, that's fine (multiple rows with same mode)
            elif mode != expected_sequence[current_mode_idx-1]:
                # We found a mode that breaks the sequence
                has_complete_sequence = False
                break
        
        # Check if we found all expected modes
        if current_mode_idx >= len(expected_sequence) - 1 and has_complete_sequence:
            valid_cycles += 1
            
            # Check if this is a reference discharge cycle
            is_reference = False
            for _, row in cycle_data[cycle_data['mode'] == -1].iterrows():
                if 'mission_type' in row and row['mission_type'] == 0:
                    is_reference = True
                    break
            
            # Create filename with reference indication if applicable
            if is_reference:
                output_path = os.path.join(output_dir, f"cycle{valid_cycles}_reference.csv")
            else:
                output_path = os.path.join(output_dir, f"cycle{valid_cycles}.csv")
                
            cycle_data.to_csv(output_path, index=False)
            cycle_type = "reference" if is_reference else "regular"
            print(f"Cycle {valid_cycles} ({cycle_type}) written to {output_path} (rows {start_idx} to {end_idx-1})")
        else:
            print(f"Skipping incomplete cycle {cycle_idx+1} (rows {start_idx} to {end_idx-1})")
    
    print(f"Processed {len(cycle_start_indices)} potential cycles, saved {valid_cycles} valid cycles")


def main():
    datasets = [
        # Regular Alt Batteries
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery00.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery00"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery01.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery01"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery10.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery10"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery11.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery11"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery20.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery20"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery21.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery21"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery22.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery22"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery23.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery23"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery30.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery30"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery31.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery31"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery40.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery40"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery41.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery41"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery50.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery50"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery51.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery51"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery52.csv", destination_directory= "../../Dataset_cycles_v2/regular_alt_batteries/battery52"),

        # Recommissioned Batteries
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery02.csv", destination_directory= "../../Dataset_cycles_v2/recommissioned_batteries/battery02"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery03.csv", destination_directory= "../../Dataset_cycles_v2/recommissioned_batteries/battery03"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery12.csv", destination_directory= "../../Dataset_cycles_v2/recommissioned_batteries/battery12"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery24.csv", destination_directory= "../../Dataset_cycles_v2/recommissioned_batteries/battery24"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery25.csv", destination_directory= "../../Dataset_cycles_v2/recommissioned_batteries/battery25"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery32.csv", destination_directory= "../../Dataset_cycles_v2/recommissioned_batteries/battery32"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery33.csv", destination_directory= "../../Dataset_cycles_v2/recommissioned_batteries/battery33"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery53.csv", destination_directory= "../../Dataset_cycles_v2/recommissioned_batteries/battery53"),

        # Second Life Batteries
        DatasetFile(source_path= "../../Datasets/second_life_batteries/battery13.csv", destination_directory= "../../Dataset_cycles_v2/second_life_batteries/battery13"),
        DatasetFile(source_path= "../../Datasets/second_life_batteries/battery36.csv", destination_directory= "../../Dataset_cycles_v2/second_life_batteries/battery36"),
        DatasetFile(source_path= "../../Datasets/second_life_batteries/battery54.csv", destination_directory= "../../Dataset_cycles_v2/second_life_batteries/battery54"),
    ]
    
    # Process each dataset
    for dataset in datasets:
        if os.path.exists(dataset.source_path):
            try:
                process_battery_file(dataset)
                print(f"Successfully processed {dataset.source_path}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing {dataset.source_path}: {str(e)}")
        else:
            print(f"Warning: File {dataset.source_path} does not exist. Skipping...")


if __name__ == "__main__":
    main()