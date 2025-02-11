import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point
import os

# Define paths
root_path = "/Users/katerinaargyriparoni/data/Tallinn/LST_2020_clipped"
urban_shapefile_path = "/Users/katerinaargyriparoni/data/Tallinn/urban_tallinn.shp"
rural_shapefile_path = "/Users/katerinaargyriparoni/data/Tallinn/rural_tallinn.shp"

# Load geometries from the shapefiles
urban_geometries = gpd.read_file(urban_shapefile_path)
rural_geometries = gpd.read_file(rural_shapefile_path)

# Create a list to store results
results = []

# Function to create random points within the polygon
def random_points_in_bounds(polygon, num_points):
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < num_points:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        new_point = Point(x, y)
        if new_point.within(polygon):
            points.append(new_point)
    return points

# Create random points for urban and rural areas just once
urban_points_all = []  # List for urban random points
rural_points_all = []  # List for rural random points

# Generate urban and rural points (outside the loop)
for geom in urban_geometries.geometry:
    urban_points = random_points_in_bounds(geom, 1000)
    urban_points_all.extend(urban_points)  # Store urban points

for geom in rural_geometries.geometry:
    rural_points = random_points_in_bounds(geom, 1000)
    rural_points_all.extend(rural_points)  # Store rural points

# Now loop over the .tif files
for dirpath, dirnames, filenames in os.walk(root_path):  # Updated to search directly in the root_path
    for file in filenames:
        if file.endswith('.tif'):
            image_path = os.path.join(dirpath, file)
            print(f"\nProcessing: {image_path}")  # Display the current image

            # Load the image using the correct method
            with rasterio.open(image_path) as src:
                scale = src.scales[0] if src.scales else 1.0
                offset = src.offsets[0] if src.offsets else 0.0
                img = src.read(1).astype(src.dtypes[0])  # Read the image
                img = img * scale + offset  # Apply scaling and offset
                img[img == -32768] = np.nan  # Replace no-data values with NaN

                # urban areas
                urban_values = [
                    img[src.index(pt.x, pt.y)] for pt in urban_points_all if not np.isnan(img[src.index(pt.x, pt.y)])
                ]
                if urban_values:
                    urban_mean = np.mean(urban_values)
                    urban_max = np.max(urban_values)
                else:
                    urban_mean, urban_max = np.nan, np.nan


                # rural areas
                rural_values = [
                    img[src.index(pt.x, pt.y)] for pt in rural_points_all if not np.isnan(img[src.index(pt.x, pt.y)])
                ]
                if rural_values:
                    rural_mean = np.mean(rural_values)
                    rural_max = np.max(rural_values)
                else:
                    rural_mean, rural_std = np.nan, np.nan



                UHI = urban_mean - rural_mean
                UHI_max = urban_max - rural_max


                results.append({
                    "image": image_path,
                    "UHI": UHI,
                    "UHI_max": UHI_max,
                    "urban_mean": urban_mean,
                    "urban_max": urban_max,
                    "rural_mean": rural_mean,
                    "rural_max": rural_max
                })

# Convert to DataFrame for easy analysis
results_df = pd.DataFrame(results)

# Define path for exporting the CSV
output_csv_path = os.path.join(root_path, "results_summary_2020_tallinn.csv")  # Updated path for saving results
results_df.to_csv(output_csv_path, index=False)

print("\nContents of the Results DataFrame:")
print(results_df)  # Print the contents of the results DataFrame