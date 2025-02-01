import json
import csv

# Function to get node latitude and longitude from the JSON data
import csv
import json

# Function to get latitude and longitude for a given node_id
def get_node_lat_lon(node_id, data):
    for element in data['elements']:
        if element['type'] == 'node' and element['id'] == node_id:
            return element['lat'], element['lon']
    return None, None

# Open the JSON file with utf-8 encoding
with open('export.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare CSV output
csv_filename = 'bridges_coordinates.csv'

with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['bridge_id', 'latitude', 'longitude']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write header row
    writer.writeheader()

    # Process the JSON data to extract bridges and their coordinates
    for element in data['elements']:
        if element['type'] == 'way' and 'tags' in element and 'bridge' in element['tags']:
            # Process each node in the bridge (assuming node IDs are listed)
            for node_id in element['nodes']:
                node_lat, node_lon = get_node_lat_lon(node_id, data)

                if node_lat and node_lon:
                    # Write the bridge ID and node coordinates to the CSV
                    writer.writerow({'bridge_id': element['id'], 'latitude': node_lat, 'longitude': node_lon})



print(f"CSV file '{csv_filename}' has been created.")

