import os
import sys
import time
import subprocess
import importlib.util

def import_module_from_file(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    creating_hdf5_path = os.path.join(base_dir, "creating_hdf5.py")
    converting_img_path = os.path.join(base_dir, "converting_img.py")
    
    # Check if required files exist
    if not os.path.exists(creating_hdf5_path):
        print(f"Error: {creating_hdf5_path} not found!")
        return
    
    if not os.path.exists(converting_img_path):
        print(f"Error: {converting_img_path} not found!")
        return
    
    # Check if download.txt exists
    download_txt_path = os.path.join(base_dir, "download.txt")
    if not os.path.exists(download_txt_path):
        print(f"Error: {download_txt_path} not found! This file is required for downloading HDF5 files.")
        return
    
    # Step 1: Download HDF5 files
    print_section_header("STEP 1: DOWNLOADING HDF5 FILES")
    print("Starting download of HDF5 files...")
    
    try:
        # Method 1: Import and run the module directly
        creating_hdf5 = import_module_from_file("creating_hdf5", creating_hdf5_path)
        print("HDF5 files download completed successfully.")
    except Exception as e:
        print(f"Error during HDF5 download: {str(e)}")
        
        # Method 2: Fall back to subprocess if import fails
        try:
            print("Trying alternative method to download HDF5 files...")
            result = subprocess.run([sys.executable, creating_hdf5_path], 
                                   check=True, text=True, capture_output=True)
            print(result.stdout)
            print("HDF5 files download completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during HDF5 download: {e}")
            print(f"Output: {e.output}")
            print(f"Error output: {e.stderr}")
            return
    
    # Check if HDF5 directory exists and has files
    hdf5_dir = os.path.join(base_dir, "HDF5")
    if not os.path.exists(hdf5_dir) or not os.listdir(hdf5_dir):
        print("Error: No HDF5 files were downloaded or the HDF5 directory is empty.")
        return
    
    # Step 2: Generate visualizations
    print_section_header("STEP 2: GENERATING VISUALIZATIONS")
    print("Starting generation of visualizations for all attributes...")
    
    try:
        # Method 1: Import and run the module directly
        converting_img = import_module_from_file("converting_img", converting_img_path)
        print("Visualization generation completed successfully.")
    except Exception as e:
        print(f"Error during visualization generation: {str(e)}")
        
        # Method 2: Fall back to subprocess if import fails
        try:
            print("Trying alternative method to generate visualizations...")
            result = subprocess.run([sys.executable, converting_img_path], 
                                   check=True, text=True, capture_output=True)
            print(result.stdout)
            print("Visualization generation completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during visualization generation: {e}")
            print(f"Output: {e.output}")
            print(f"Error output: {e.stderr}")
            return
    
    # Check if visualization output directory exists
    vis_output_dir = os.path.join(base_dir, "visualization_output")
    if not os.path.exists(vis_output_dir):
        print("Warning: Visualization output directory was not created.")
        return
    
    # Count the number of generated images
    total_images = 0
    for root, dirs, files in os.walk(vis_output_dir):
        total_images += len([f for f in files if f.endswith('.png')])
    
    # Final summary
    print_section_header("PROCESS COMPLETED")
    print(f"Total HDF5 files processed: {len(os.listdir(hdf5_dir))}")
    print(f"Total visualization images generated: {total_images}")
    print(f"Visualization categories: {', '.join(os.listdir(vis_output_dir))}")
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")