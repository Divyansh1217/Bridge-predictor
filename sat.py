import os
import ee
import requests
import cv2
import numpy as np
import pandas as pd

# Initialize Google Earth Engine (GEE)
ee.Initialize(project='ee-divyansh27aggarwal')

# Create folder for saving images
IMAGE_DIR = "bridges_sat_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# CSV file path
CSV_FILE = os.path.join(IMAGE_DIR, "bridges_dataset_images.csv")

# Define bridge locations (lat, lon, ID)
bridges = [
    {"id": "24515425", "lat": 28.4016, "lon": 77.1405},
    {"id": "24563335", "lat": 28.4598778, "lon": 77.0748951},
    {"id": "25878294", "lat": 28.4608778, "lon": 77.0758951},
]

# Time ranges
time_ranges = {
    "old": ("2013-01-01", "2015-12-31"),
    "middle": ("2017-01-01", "2019-12-31"),
    "new": ("2022-01-01", "2024-12-31")
}

# Function to fetch and download satellite images
def fetch_satellite_image(lat, lon, bridge_id, time_label, date_range):
    start_date, end_date = date_range

    # First try Sentinel-2
    image_collection = (
        ee.ImageCollection("COPERNICUS/S2")  # Sentinel-2 TOA dataset
        .filterBounds(ee.Geometry.Point(lon, lat))
        .filterDate(start_date, end_date)
        .sort("CLOUDY_PIXEL_PERCENTAGE")  # Least cloudy images first
    )

    image = image_collection.first()

    if image.getInfo() is None:
        print(f"No Sentinel-2 images for {bridge_id} at {lat}, {lon} during {time_label}, trying Landsat...")

        # Try Landsat-8 Collection 2 Level 2
        image_collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")  # Corrected dataset ID
            .filterBounds(ee.Geometry.Point(lon, lat))
            .filterDate(start_date, end_date)
            .sort("CLOUD_COVER")
        )
        image = image_collection.first()

        if image.getInfo() is None:
            print(f"No Landsat images found for {bridge_id} at {lat}, {lon} during {time_label}. Skipping...")
            return None

    # Select RGB Bands (Red, Green, Blue) for True Color Visualization
    image = image.select(['B4', 'B3', 'B2'])

    # Normalize Pixel Values (Avoids White Image Issue)
    image = image.unitScale(0, 3000).multiply(255).toByte()

    # Increase image area (2 km buffer)
    region = ee.Geometry.Point(lon, lat).buffer(2000).bounds()

    # Download URL (TIFF format)
    try:
        url = image.getDownloadURL({"scale": 10, "region": region, "format": "GeoTIFF"})
        print(f"Generated URL: {url}")
        
        # Download the image
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            image_path = os.path.join(IMAGE_DIR, f"{bridge_id}_{time_label}.tif")
            with open(image_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Image saved: {image_path}")
            return image_path
        else:
            print(f"Failed to download image for {bridge_id} ({time_label}). HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Failed to generate URL for {bridge_id} ({time_label}): {str(e)}")
        return None

# Function to enhance image contrast using OpenCV
def enhance_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # Convert to grayscale (if necessary)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img = clahe.apply(img)

        # Save enhanced image
        enhanced_path = image_path.replace(".tif", "_enhanced.tif")
        cv2.imwrite(enhanced_path, img)
        print(f"Enhanced image saved: {enhanced_path}")
        return enhanced_path
    except Exception as e:
        print(f"Error enhancing image {image_path}: {str(e)}")
        return None

# Create dataset
dataset = []

for bridge in bridges:
    bridge_id = bridge["id"]
    lat, lon = bridge["lat"], bridge["lon"]

    image_paths = {}
    for time_label, date_range in time_ranges.items():
        img_path = fetch_satellite_image(lat, lon, bridge_id, time_label, date_range)
        if img_path:
            enhanced_path = enhance_image(img_path)
        else:
            enhanced_path = "NA"
        image_paths[time_label] = enhanced_path

    dataset.append({
        "Latitude": lat,
        "Longitude": lon,
        "Old Image Path": image_paths["old"],
        "Middle Image Path": image_paths["middle"],
        "New Image Path": image_paths["new"]
    })

# Save to CSV
df = pd.DataFrame(dataset)
df.to_csv(CSV_FILE, index=False)
print(f"Dataset saved to {CSV_FILE}")
