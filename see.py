import rasterio
import matplotlib.pyplot as plt

# Load and display an image
image_path = "bridges_sat_images/24515425_old.tif"

with rasterio.open(image_path) as src:
    image = src.read([1, 2, 3])  # Read RGB bands
    plt.imshow(image.transpose(1, 2, 0))  # Transpose for correct shape
    plt.title("Bridge Satellite Image")
    plt.show()
