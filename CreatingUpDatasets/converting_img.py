import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re

# Define input folder
input_folder = "HDF5Up"

# Define target years to process (2020-2024)
target_years = list(range(2020, 2025))  # [2020, 2021, 2022, 2023, 2024]

# Define attributes to visualize based on your selected parameters
attributes = {
    # Soil moisture content (0-100cm)
    "SoilM_0_100cm": {"title": "Soil Moisture Content (0-100cm)", "vmin": 0, "vmax": 500, "cmap": "YlGnBu", "unit": "kg m⁻²"},
    
    # Liquid precipitation (rainfall)
    "Rainf": {"title": "Liquid Precipitation (Rainfall)", "vmin": 0, "vmax": 0.001, "cmap": "Blues", "unit": "kg m⁻²"},
    
    # Potential evapotranspiration
    "PotEvap": {"title": "Potential Evapotranspiration", "vmin": 0, "vmax": 300, "cmap": "YlOrRd", "unit": "W m⁻²"},
    
    # Average surface skin temperature
    "AvgSurfT": {"title": "Average Surface Skin Temperature", "vmin": 260, "vmax": 320, "cmap": "RdYlBu_r", "unit": "K"},
    
    # Soil moisture availability (0-100cm)
    "SMAvail_0_100cm": {"title": "Soil Moisture Availability (0-100cm)", "vmin": 0, "vmax": 100, "cmap": "GnBu", "unit": "%"},
}

# Create main output folder with year-specific subfolder
main_output_folder = "visualization_output_2020_2024"
os.makedirs(main_output_folder, exist_ok=True)

# Create a folder for each attribute
for attr in attributes:
    attr_folder = os.path.join(main_output_folder, attr)
    os.makedirs(attr_folder, exist_ok=True)

# Function to extract year from filename
def extract_year_from_filename(filename):
    # Assuming format like "A20200102.HDF5" where 2020 is the year
    year_match = re.search(r'A(\d{4})\d{4}\.HDF5', filename)
    if year_match:
        return int(year_match.group(1))
    return None

# Count files for statistics
total_files = 0
processed_files = 0
files_by_year = {year: 0 for year in target_years}

# First count total files and files per target year
for file in os.listdir(input_folder):
    if file.endswith(".HDF5"):
        total_files += 1
        year = extract_year_from_filename(file)
        if year in target_years:
            files_by_year[year] += 1

print(f"Found {total_files} total HDF5 files")
print(f"Files to process by year: {files_by_year}")

# Loop through all .HDF5 files in the input folder
for file in sorted(os.listdir(input_folder)):
    if file.endswith(".HDF5"):
        # Extract year from filename to filter
        year = extract_year_from_filename(file)
        
        # Skip files that don't match our target years (2020-2024)
        if year not in target_years:
            continue
            
        file_path = os.path.join(input_folder, file)
        processed_files += 1
        print(f"\nProcessing file: {file} ({processed_files}/{sum(files_by_year.values())})")

        # Open HDF5 File
        with h5py.File(file_path, 'r') as data:
            # Extract latitude and longitude arrays from the HDF5 file
            lats = data['/lat'][:]  # Latitude array
            lons = data['/lon'][:]  # Longitude array

            # Ensure latitude is in ascending order
            if lats[0] > lats[-1]:
                lats = np.flip(lats)
                flip_lat = True
            else:
                flip_lat = False

            # Process each attribute
            for attr_name, attr_props in attributes.items():
                # Check if the attribute exists in the file
                attr_path = f'/{attr_name}'
                if attr_path not in data:
                    print(f"  Attribute {attr_name} not found in {file}, skipping...")
                    continue

                print(f"  Visualizing {attr_name}...")
                
                # Load data
                attr_data = data[attr_path][:]
                
                # Remove first dimension if exists and handle dimensions
                if len(attr_data.shape) > 2:
                    attr_data = attr_data[0, :, :]
                
                # Flip data if needed
                if flip_lat:
                    attr_data = np.flip(attr_data, axis=0)
                
                # Mask missing values (if any)
                attr_data = np.ma.masked_where(attr_data < -9990, attr_data)
                
                # Create a plot with Cartopy
                fig = plt.figure(figsize=(12, 8))
                ax = plt.axes(projection=ccrs.PlateCarree())
                
                # Plot data using pcolormesh
                im = ax.pcolormesh(lons, lats, attr_data, 
                                  vmin=attr_props["vmin"], 
                                  vmax=attr_props["vmax"], 
                                  cmap=attr_props["cmap"], 
                                  transform=ccrs.PlateCarree())
                
                # Add geographic features
                ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
                ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
                ax.add_feature(cfeature.STATES, linestyle=':', linewidth=0.3)
                
                # Add gridlines and labels
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                 linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
                cbar.set_label(f"{attr_props['title']} ({attr_props['unit']})", fontsize=10)
                
                # Add title with date
                plt.title(f"{attr_props['title']}\nDate: {file[1:5]}-{file[5:7]}-{file[7:9]}", fontsize=12)
                
                # Create year-organized folders
                year_folder = os.path.join(main_output_folder, attr_name, str(year))
                os.makedirs(year_folder, exist_ok=True)
                
                # Save the plot in the appropriate year folder
                output_file = os.path.join(year_folder, f"{attr_name}_{file[1:9]}.png")
                plt.savefig(output_file, dpi=200, bbox_inches='tight')
                print(f"    Saved: {output_file}")
                
                # Close the figure to free memory
                plt.close(fig)
                
        # Force garbage collection to free memory
        import gc
        gc.collect()

print(f"\nVisualization complete! Processed {processed_files} files from years {target_years}")
print(f"Output saved in: {main_output_folder}")