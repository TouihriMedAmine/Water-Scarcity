import os
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define input folder
input_folder = "HDF5Up2020-2024"

# Define attributes to visualize
attributes = {
    "Evap": {"title": "Evaporation", "vmin": 0, "vmax": 0.01, "cmap": "YlGnBu"},
    "Rainf": {"title": "Rainfall Rate", "vmin": 0, "vmax": 0.01, "cmap": "Blues"},
    "SoilM_0_10cm": {"title": "Soil Moisture (0-10 cm)", "vmin": 0, "vmax": 0.5, "cmap": "BrBG"},
    "RootMoist": {"title": "Root Zone Moisture", "vmin": 0, "vmax": 0.5, "cmap": "YlOrBr"},
    "TVeg": {"title": "Vegetation Transpiration", "vmin": 0, "vmax": 0.01, "cmap": "Greens"},
}

# Create output folder
main_output_folder = "visualization_output_nc"
os.makedirs(main_output_folder, exist_ok=True)

# Create folders for each attribute
for attr in attributes:
    os.makedirs(os.path.join(main_output_folder, attr), exist_ok=True)

# Loop through all .nc files
for file in os.listdir(input_folder):
    if file.endswith(".nc"):
        file_path = os.path.join(input_folder, file)
        print(f"Processing file: {file}")

        # Open NetCDF file
        with Dataset(file_path, 'r') as nc_file:
            # Attempt common names for lat/lon
            lat_names = ['lat', 'latitude']
            lon_names = ['lon', 'longitude']

            for name in lat_names:
                if name in nc_file.variables:
                    lats = nc_file.variables[name][:]
                    break
            else:
                print(f"  Latitude not found in {file}, skipping...")
                continue

            for name in lon_names:
                if name in nc_file.variables:
                    lons = nc_file.variables[name][:]
                    break
            else:
                print(f"  Longitude not found in {file}, skipping...")
                continue

            # Ensure latitude is ascending
            if lats[0] > lats[-1]:
                lats = np.flip(lats)
                flip_lat = True
            else:
                flip_lat = False

            # Loop over each variable
            for attr_name, attr_props in attributes.items():
                if attr_name not in nc_file.variables:
                    print(f"  Attribute {attr_name} not found in {file}, skipping...")
                    continue

                print(f"  Visualizing {attr_name}...")

                # Read the data
                attr_data = nc_file.variables[attr_name][:]
                if attr_data.ndim > 2:
                    attr_data = attr_data[0, :, :]

                if flip_lat:
                    attr_data = np.flip(attr_data, axis=0)

                # Mask invalid values (assuming -9999 is a fill value)
                attr_data = np.ma.masked_where(attr_data < -9990, attr_data)

                # Plotting
                fig = plt.figure(figsize=(10, 6))
                ax = plt.axes(projection=ccrs.PlateCarree())

                im = ax.pcolormesh(lons, lats, attr_data,
                                   vmin=attr_props["vmin"],
                                   vmax=attr_props["vmax"],
                                   cmap=attr_props["cmap"],
                                   transform=ccrs.PlateCarree())

                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False

                cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
                cbar.set_label(attr_props["title"])

                plt.title(f"{attr_props['title']} - {file.replace('.nc', '')}")

                output_file = os.path.join(main_output_folder, attr_name, file.replace(".nc", ".png"))
                plt.savefig(output_file, dpi=200, bbox_inches='tight')
                print(f"    Saved: {output_file}")
                plt.close(fig)

print("Visualization complete!")
