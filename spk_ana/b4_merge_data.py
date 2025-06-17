import os
import shutil
from pathlib import Path

def merge_folders(folder_1, folder_2, output_path):
    """
    Merge the contents of two folders into a new output folder.
    
    Args:
        folder_1 (str): Path to first source folder
        folder_2 (str): Path to second source folder  
        output_path (str): Path to destination folder
    """
    # Convert to Path objects for easier manipulation
    folder1 = Path(folder_1)
    folder2 = Path(folder_2)
    output = Path(output_path)
    
    # Create output directory if it doesn't exist
    output.mkdir(parents=True, exist_ok=True)
    
    # Get all subdirectories from both folders
    subfolders1 = {item.name for item in folder1.iterdir() if item.is_dir()}
    subfolders2 = {item.name for item in folder2.iterdir() if item.is_dir()}
    
    # Get all unique subfolder names
    all_subfolders = subfolders1.union(subfolders2)
    
    for subfolder_name in all_subfolders:
        src1 = folder1 / subfolder_name
        src2 = folder2 / subfolder_name
        dest = output / subfolder_name
        
        # Create destination subfolder
        dest.mkdir(exist_ok=True)
        
        # Copy files from folder1 if subfolder exists there
        if src1.exists():
            for item in src1.iterdir():
                if item.is_file():
                    shutil.copy2(item, dest / item.name)
        
        # Copy files from folder2 if subfolder exists there
        if src2.exists():
            for item in src2.iterdir():
                if item.is_file():
                    # Handle potential name conflicts
                    dest_file = dest / item.name
                    if dest_file.exists():
                        # Add suffix to avoid overwriting
                        stem = item.stem
                        suffix = item.suffix
                        counter = 1
                        while dest_file.exists():
                            dest_file = dest / f"{stem}_{counter}{suffix}"
                            counter += 1
                    shutil.copy2(item, dest_file)


if __name__ == "__main__":
    # Replace these paths with your actual folder paths
    pn_root = r"\\NJJK-NAS\visual\66_paper\MANUSCRIPT\20250610-v6-submit\figShare_upload\DATA"
    # bin_size = 50 # bin size in ms
    bin_size = 500 # bin size in ms

    folder_1 = os.path.join(pn_root, f'04_result_{bin_size}ms', 'b3_task_raster','VM20','20231010','01_driving')
    folder_2 = os.path.join(pn_root, f'04_result_{bin_size}ms', 'b3_task_raster','VM23','20231108','01_driving')
    destination = os.path.join(pn_root, f'04_result_{bin_size}ms', 'b3_task_raster','VM20_VM23')
    
    merge_folders(folder_1, folder_2, destination)
    print(f"Folders merged successfully into: {destination}")