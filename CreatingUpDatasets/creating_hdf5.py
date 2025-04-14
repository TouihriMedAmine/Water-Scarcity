import requests
import os
import re
import random
from datetime import datetime

# Read the links from the file
hdflinks = open('downloadUp.txt').readlines()

# Clean links for breakline characters
hdflinks = [re.sub('\n', '', link) for link in hdflinks]

# Target years
target_years = [2020, 2021, 2022, 2023, 2024]

# Function to extract year from link
def extract_year_from_link(link):
    # Pattern for NLDAS links: matches year in formats like A20201231 or .../2020/...
    # First try the direct filename pattern
    filename_match = re.search(r'A(\d{4})\d{4}', link)
    if filename_match:
        return int(filename_match.group(1))
    
    # Then try the directory pattern
    dir_match = re.search(r'/(\d{4})/', link)
    if dir_match:
        return int(dir_match.group(1))
    
    return None

# Filter links by target years
filtered_links = []
for link in hdflinks:
    year = extract_year_from_link(link)
    if year and year in target_years:
        filtered_links.append(link)

print(f"Total links: {len(hdflinks)}")
print(f"Filtered links for years {target_years}: {len(filtered_links)}")

# Create the main directory if it doesn't exist
os.makedirs('./HDF5Up2020-2024', exist_ok=True)

# Group links by 10 and select one random link from each group
selected_links = []
for i in range(0, len(filtered_links), 10):
    group = filtered_links[i:i+10]
    if group:  # Make sure the group is not empty
        selected_link = random.choice(group)
        selected_links.append(selected_link)

# Total number of files to download
total_files = len(selected_links)
print(f"Selected {total_files} links to download (1 per 10 links)")

# Counter to keep track of downloaded files
downloaded_count = 0

# Loop through each selected link
for link in selected_links:
    # Extract the date pattern using regex
    date_pattern = re.search(r'A\d{8}', link)
    if date_pattern:
        date_str = date_pattern.group(0)
        # Create filename in the main directory
        filename = f"./HDF5Up2020-2024/{date_str}.nc"
    else:
        # Fallback filename if date pattern is not found
        filename = f"./HDF5Up2020-2024/nldas_{datetime.now().strftime('%Y%m%d%H%M%S')}.nc"
        print(f"Could not extract date from link: {link}")
        print(f"Using fallback filename: {filename}")
    
    print(f"Processing: {link}")
    
    # Check if the file already exists
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping download.")
        print(f"Progress: {downloaded_count}/{total_files} downloaded")
        downloaded_count += 1
        continue
    
    # If the file doesn't exist, proceed with the download
    try:
        with requests.Session() as session:
            req = session.request('get', link)
            r = session.get(req.url, auth=('maher123', 'Pppppppppp0@'))
            
            if r.status_code == 200:
                # Save the file
                with open(filename, 'wb') as f:
                    f.write(r.content)
                downloaded_count += 1
                print(f"Downloaded {filename}")
                print(f"Progress: {downloaded_count}/{total_files} downloaded")
            else:
                print(f"Failed to download {link}. Status code: {r.status_code}")
    except Exception as e:
        print(f"Error downloading {link}: {str(e)}")

print(f"Download complete. Total files downloaded: {downloaded_count}/{total_files}")

# Print summary of downloaded files
downloaded_files = [f for f in os.listdir('./HDF5Up2020-2024') if f.endswith('.nc')]
print(f"\nTotal files downloaded in HDF5Up2020-2024 directory: {len(downloaded_files)}")