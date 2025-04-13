import os
import subprocess
import time

def run_script(script_path, description):
    print(f"\n{'='*50}")
    print(f"Starting {description}...")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    try:
        subprocess.run(['python', script_path], check=True)
        execution_time = time.time() - start_time
        print(f"\n{description} completed successfully!")
        print(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
        return True
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        print(f"\nError during {description}: {e}")
        print(f"Time until error: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
        return False

def main():
    total_start_time = time.time()
    
    # Define paths
    creating_hdf5_path = os.path.join('Tassounim', 'creating_hdf5.py')
    converting_img_path = 'converting_img.py'
    
    # Step 1: Download HDF5 files
    if not run_script(creating_hdf5_path, "HDF5 files download"):
        print("Failed to download HDF5 files. Stopping execution.")
        return
    
    # Wait a moment to ensure all files are properly saved
    time.sleep(2)
    
    # Step 2: Convert HDF5 to images
    if not run_script(converting_img_path, "Image conversion"):
        print("Failed to convert images.")
        return
    
    total_execution_time = time.time() - total_start_time
    print("\nAll operations completed successfully!")
    print(f"Total execution time: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()