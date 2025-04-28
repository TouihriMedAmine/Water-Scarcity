import requests
import os
import re
import random

# Read the links from the file
hdflinks = open('downloadUp.txt').readlines()

# Clean links for breakline characters
hdflinks = [re.sub('\n', '', link) for link in hdflinks]

# Create directory if it doesn't exist
os.makedirs('./HDF5Up', exist_ok=True)

# Counter to keep track of downloaded files
downloaded_count = 0

# Group links by 10 and select one random link from each group
selected_links = []
for i in range(0, len(hdflinks), 10):
    group = hdflinks[i:i+10]
    if group:  # Make sure the group is not empty
        selected_link = random.choice(group)
        selected_links.append(selected_link)

# Total number of files to download
total_files = len(selected_links)

# Loop through each selected link
for link in selected_links:
    print(link)
    
    # Extract the date pattern using regex
    date_pattern = re.search(r'A\d{8}', link)
    if date_pattern:
        date_str = date_pattern.group(0)
        filename = f"./HDF5Up/{date_str}.HDF5"
    else:
        print(f"Could not extract date from link: {link}")
        continue
    
    # Check if the file already exists
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping download.")
        print(f"{downloaded_count}/{total_files} downloaded")
        downloaded_count += 1
        continue
    
    # If the file doesn't exist, proceed with the download
    with requests.Session() as session:
        req = session.request('get', link)
        r = session.get(req.url, auth=('baklouti.linda', 'LILI123@linda'))
        
        # Save the file
        with open(filename, 'wb') as f:
            f.write(r.content)
        downloaded_count += 1
        print(f"Downloaded {filename}")
        print(f"{downloaded_count}/{total_files} downloaded")

print(f"Download complete. Total files downloaded: {downloaded_count}/{total_files}")