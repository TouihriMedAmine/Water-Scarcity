import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import os

# -- Define the base date and number of iterations
start_date = datetime(2000, 3, 1)  # Start date
num_days = 30  # Number of days to process
time_steps_per_day = 8  # 3-hourly data (24 hours / 3 hours = 8 steps)

# Calculate total number of files to process
total_files = num_days * time_steps_per_day
processed_count = 0  # Counter to track processed files

# -- Loop through each day
for day in range(num_days):
    # Generate the current date
    current_date = start_date + timedelta(days=day)

    # -- Loop through each 3-hour time step
    for time_step in range(time_steps_per_day):
        # Calculate the current hour
        current_hour = 3 * time_step  # 0, 3, 6, ..., 21

        # Generate the filename with hour included (format: YYYYMMDDHH)
        file_date = (current_date + timedelta(hours=current_hour)).strftime('%Y%m%d%H')
        file = f'HDF5/{file_date}.HDF5'  # Construct the file path

        # Check if the file exists
        if not os.path.exists(file):
            print(f"File not found: {file}. Skipping...")
            processed_count += 1  # Increment counter even if file is skipped
            print(f"{processed_count}/{total_files} processed")
            continue

        try:
            # -- Open IMERG HDF5 File
            with h5py.File(file, 'r') as data:
                # -- Load precipitation data
                precip = data['/precipitation'][:]

                # -- Mask missing values (if any)
                precip = np.ma.masked_where(precip < 0, precip)

                # -- Define the region of interest (India: lat 8°N to 37°N, lon 68°E to 97°E)
                lat_min, lat_max = 30, 38
                lon_min, lon_max = 7, 12

                # -- Convert lat/lon bounds to indices
                # Assuming the data is on a 0.1° grid (IMERG resolution)
                lat = np.linspace(-90, 90, precip.shape[0])
                lon = np.linspace(-180, 180, precip.shape[1])

                lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
                lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]

                # -- Subset the data
                precip_region = precip[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]

                # -- Create a plot with Cartopy
                fig = plt.figure(figsize=(10, 6))
                ax = plt.axes(projection=ccrs.PlateCarree())  # Use PlateCarree projection

                # -- Plot precipitation data
                im = ax.imshow(precip_region, norm=LogNorm(vmin=0.1, vmax=100),
                            extent=[lon_min, lon_max, lat_min, lat_max],
                            cmap='Blues', origin='upper', transform=ccrs.PlateCarree())

                # -- Add geographic features
                ax.add_feature(cfeature.COASTLINE)  # Add coastlines
                ax.add_feature(cfeature.BORDERS, linestyle=':')  # Add country borders

                # -- Add gridlines and labels
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False  # Disable top labels
                gl.right_labels = False  # Disable right labels

                # -- Add color bar
                cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.7)
                cbar.set_label('Precipitation (mm/hr)')

                # -- Add title
                plt.title(f'IMERG Precipitation ({current_date.strftime("%d %b %Y")}, 00:00-02:59 UTC)\n'
                        f'Region: {lat_min}°N to {lat_max}°N, {lon_min}°E to {lon_max}°E')

                # -- Save the plot
                output_file = f'./images/img_{file_date}.png'
                plt.savefig(output_file, dpi=200, bbox_inches='tight')
                plt.close()  # Close the plot to free memory

                print(f"Processed and saved: {output_file}")
                processed_count += 1  # Increment counter after successful processing
                print(f"{processed_count}/{total_files} processed")

        except FileNotFoundError:
            print(f"File not found: {file}. Skipping...")
            processed_count += 1  # Increment counter even if file is skipped
            print(f"{processed_count}/{total_files} processed")
        
        except Exception as e:
            print(f"Error processing {file}: {e}")
            processed_count += 1  # Increment counter even if there's an error
            print(f"{processed_count}/{total_files} processed")

print(f"Processing complete. Total files processed: {processed_count}/{total_files}")