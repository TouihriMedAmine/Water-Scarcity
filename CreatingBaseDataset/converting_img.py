import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define input folder
input_folder = "HDF5"

# Define attributes to visualize based on README.md
attributes = {
    # Plant Water Consumption
    "Evap_tavg": {"title": "Evapotranspiration", "vmin": 0, "vmax": 10, "cmap": "Blues"},
    "Tveg_tavg": {"title": "Plant Transpiration", "vmin": 0, "vmax": 8, "cmap": "Greens"},
    "ESoil_tavg": {"title": "Soil Evaporation", "vmin": 0, "vmax": 5, "cmap": "YlOrBr"},
    "PotEvap_tavg": {"title": "Potential Evaporation", "vmin": 0, "vmax": 15, "cmap": "plasma"},
    
    # Soil Water Retention
    "SoilMoist_S_tavg": {"title": "Soil Moisture", "vmin": 5, "vmax": 45, "cmap": "turbo"},
    "SoilMoi0_10cm_inst": {"title": "Soil Moisture (0-10cm)", "vmin": 0, "vmax": 50, "cmap": "Blues_r"},
    "RootMoist_inst": {"title": "Root Zone Soil Moisture", "vmin": 0, "vmax": 50, "cmap": "GnBu"},
    
    # Irrigation Demand
    "Tair_f_inst": {"title": "Air Temperature", "vmin": 270, "vmax": 310, "cmap": "coolwarm"},
    "SWdown_f_tavg": {"title": "Solar Radiation", "vmin": 0, "vmax": 300, "cmap": "YlOrRd"},
    "Rainf_tavg": {"title": "Precipitation", "vmin": 0, "vmax": 0.0001, "cmap": "Blues"},
}

# Create main output folder
main_output_folder = "visualization_output"
os.makedirs(main_output_folder, exist_ok=True)

# Create a folder for each attribute
for attr in attributes:
    attr_folder = os.path.join(main_output_folder, attr)
    os.makedirs(attr_folder, exist_ok=True)

# Loop through all .HDF5 files in the input folder
for file in os.listdir(input_folder):
    if file.endswith(".HDF5"):
        file_path = os.path.join(input_folder, file)
        print(f"Processing file: {file}")

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
                attr_data = np.ma.masked_where(attr_data < -9990, attr_data)  # Assuming -9999 is a common missing value
                
                # Create a plot with Cartopy
                fig = plt.figure(figsize=(10, 6))
                ax = plt.axes(projection=ccrs.PlateCarree())
                
                # Plot data using pcolormesh
                im = ax.pcolormesh(lons, lats, attr_data, 
                                  vmin=attr_props["vmin"], 
                                  vmax=attr_props["vmax"], 
                                  cmap=attr_props["cmap"], 
                                  transform=ccrs.PlateCarree())
                
                # Add geographic features
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                
                # Add gridlines and labels
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                 linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
                cbar.set_label(attr_props["title"])
                
                # Set map boundaries for Tunisia region
                ax.set_xlim([7, 12])  # Longitude limits
                ax.set_ylim([30, 38])  # Latitude limits
                
                # Add title
                plt.title(f"{attr_props['title']} - {file.replace('.HDF5', '')}")
                
                # Save the plot in the appropriate folder
                output_file = os.path.join(main_output_folder, attr_name, file.replace(".HDF5", ".png"))
                plt.savefig(output_file, dpi=200, bbox_inches='tight')
                print(f"    Saved: {output_file}")
                
                # Close the figure to free memory
                plt.close(fig)

print("Visualization complete!")