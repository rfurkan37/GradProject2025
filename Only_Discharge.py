import pandas as pd
import os
from dataclasses import dataclass
from typing import List


@dataclass
class DatasetFile:
    source_path: str
    destination_directory: str


def extract_discharge_segments(dataset: DatasetFile):
    """
    Process a single battery CSV file and extract only discharge segments to separate files.
    
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
    
    print(f"Found {len(discharge_segments)} discharge segments")
    
    if len(discharge_segments) == 0:
        print("No discharge segments found. Exiting.")
        return
    
    # Process each discharge segment
    for segment_idx, (start_idx, end_idx) in enumerate(discharge_segments, 1):
        # Extract segment data
        segment_data = df.iloc[start_idx:end_idx].copy()
        
        # Verify this is actually a discharge segment
        if segment_data['mode'].iloc[0] != -1:
            print(f"Warning: Segment {segment_idx} does not start with discharge mode. Skipping.")
            continue
            
        # Check if this is a reference discharge
        is_reference = False
        if 'mission_type' in segment_data.columns:
            # If any row in this discharge segment has mission_type == 0, it's a reference discharge
            is_reference = (segment_data['mission_type'] == 0).any()
        
        # Create filename with reference indication if applicable
        if is_reference:
            output_path = os.path.join(output_dir, f"discharge{segment_idx}_reference.csv")
        else:
            output_path = os.path.join(output_dir, f"discharge{segment_idx}.csv")
        
        # Save to CSV
        segment_data.to_csv(output_path, index=False)
        
        discharge_type = "reference" if is_reference else "regular"
        print(f"Discharge segment {segment_idx} ({discharge_type}) written to {output_path} (rows {start_idx} to {end_idx-1})")
    
    print(f"Processed and saved {len(discharge_segments)} discharge segments")


def main():
    datasets = [
        # Regular Alt Batteries
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery00.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery00"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery01.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery01"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery10.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery10"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery11.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery11"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery20.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery20"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery21.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery21"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery22.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery22"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery23.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery23"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery30.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery30"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery31.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery31"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery40.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery40"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery41.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery41"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery50.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery50"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery51.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery51"),
        DatasetFile(source_path= "../../Datasets/regular_alt_batteries/battery52.csv", destination_directory= "../../Only_Discharge_Dataset/regular_alt_batteries/battery52"),

        # Recommissioned Batteries
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery02.csv", destination_directory= "../../Only_Discharge_Dataset/recommissioned_batteries/battery02"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery03.csv", destination_directory= "../../Only_Discharge_Dataset/recommissioned_batteries/battery03"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery12.csv", destination_directory= "../../Only_Discharge_Dataset/recommissioned_batteries/battery12"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery24.csv", destination_directory= "../../Only_Discharge_Dataset/recommissioned_batteries/battery24"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery25.csv", destination_directory= "../../Only_Discharge_Dataset/recommissioned_batteries/battery25"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery32.csv", destination_directory= "../../Only_Discharge_Dataset/recommissioned_batteries/battery32"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery33.csv", destination_directory= "../../Only_Discharge_Dataset/recommissioned_batteries/battery33"),
        DatasetFile(source_path= "../../Datasets/recommissioned_batteries/battery53.csv", destination_directory= "../../Only_Discharge_Dataset/recommissioned_batteries/battery53"),

        # Second Life Batteries
        DatasetFile(source_path= "../../Datasets/second_life_batteries/battery13.csv", destination_directory= "../../Only_Discharge_Dataset/second_life_batteries/battery13"),
        DatasetFile(source_path= "../../Datasets/second_life_batteries/battery36.csv", destination_directory= "../../Only_Discharge_Dataset/second_life_batteries/battery36"),
        DatasetFile(source_path= "../../Datasets/second_life_batteries/battery54.csv", destination_directory= "../../Only_Discharge_Dataset/second_life_batteries/battery54"),
    ]
    
    # Process each dataset
    for dataset in datasets:
        if os.path.exists(dataset.source_path):
            try:
                extract_discharge_segments(dataset)
                print(f"Successfully processed {dataset.source_path}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing {dataset.source_path}: {str(e)}")
        else:
            print(f"Warning: File {dataset.source_path} does not exist. Skipping...")


if __name__ == "__main__":
    main()