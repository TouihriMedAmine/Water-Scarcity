import os
import json

# Path to the folder containing your 3 JSON files
input_folder = r"Extraction_Chat\extracted_data"
output_file = r"Extraction_Chat\merged_dataset.json"

merged_data = []

# Loop through all .json files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data.append(data)  # Each file is one chapter-like JSON object

# Save the merged data to a new JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print("✅ JSON files merged successfully into:", output_file)
