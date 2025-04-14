import requests
import os
import re

# Read the links from the file
hdflinks = open('download.txt').readlines()

# Clean links for breakline characters
hdflinks = [re.sub('\n', '', link) for link in hdflinks]

# Limit to the first 10 links (optional)
# hdflinks = hdflinks[:30]

# Create directory if it doesn't exist
os.makedirs('./HDF5', exist_ok=True)

# Counter to keep track of downloaded files
downloaded_count = 0

# Total number of files to download
total_files = len(hdflinks)

# Loop through each link
for link in hdflinks:
    print(link)
    
    # Extract the filename from the link
    imagename = link.split('.')
    filename = f"./HDF5/{imagename[7][:10]}.HDF5"
    
    # Check if the file already exists
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping download.")
        print(f"{downloaded_count}/{total_files} downloaded")
        downloaded_count += 1  # Increment the counter
        continue
    
    # If the file doesn't exist, proceed with the download
    with requests.Session() as session:
        req = session.request('get', link)
        r = session.get(req.url, auth=('maher123', 'Pppppppppp0@'))
        
        # Save the file
        with open(filename, 'wb') as f:
            f.write(r.content)
        downloaded_count += 1  # Increment the counter
        print(f"Downloaded {filename}")
        print(f"{downloaded_count}/{total_files} downloaded")

print(f"Download complete. Total files downloaded: {downloaded_count}/{total_files}")
