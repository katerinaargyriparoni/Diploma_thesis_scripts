from osgeo import gdal, ogr
import os


def clip_raster_with_shapefile(input_raster, shapefile, output_raster):
    try:
        result = gdal.Warp(
            output_raster,  # Output file
            input_raster,  # Input raster
            format='GTiff',  # Output format
            cutlineDSName=shapefile,  # Shapefile to be used as mask
            cropToCutline=True,  # Crop to the mask
            dstNodata=None,  # Remove NoData values
            creationOptions=["COMPRESS=LZW"]  # Optional: compress the output
        )

        if result is None:
            print(f"Error executing gdal.Warp for {input_raster}.")
            print(f"GDAL Error: {gdal.GetLastErrorMsg()}")
        else:
            print(f"The raster was successfully clipped. Saved as: {output_raster}")
    except Exception as e:
        print(f"Error executing gdal.Warp: {e}")


def check_shapefile(shapefile):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.Open(shapefile, 0)

    if not datasource:
        print(f"Error: Could not open the shapefile {shapefile}.")
        return False

    layer = datasource.GetLayer()
    if layer.GetFeatureCount() == 0:
        print(f"The shapefile {shapefile} contains no features.")
        return False

    print(f"The shapefile {shapefile} is valid and contains {layer.GetFeatureCount()} features.")
    return True


def check_raster(input_raster):
    ds = gdal.Open(input_raster)
    if ds is None:
        print(f"Error: Could not open the raster {input_raster}.")
        return False
    print(f"The raster {input_raster} was opened successfully.")
    return True


if __name__ == "__main__":

    shapefile_path = '/Users/katerinaargyriparoni/data/Tallinn/bounding_box_Tallinn.shp'
    input_folder_path = '/Users/katerinaargyriparoni/data/Tallinn/LST_2020'
    output_folder_path = '/Users/katerinaargyriparoni/data/Tallinn/LST_2020_clipped'


    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    if not check_shapefile(shapefile_path):
        print("Error with the shapefile, terminating the program.")
        exit()


    print("Contents of the input folder:")
    for root, dirs, files in os.walk(input_folder_path):
        print(f"Folder: {root}")
        for filename in files:
            print(f"  File: {filename}")

    # Traverse all .tif files in the folder and subfolders
    tif_files_found = False  # A flag to check if any files were found
    for root, dirs, files in os.walk(input_folder_path):
        for filename in files:
            if filename.endswith('.tif') or filename.endswith('.tiff'):  # Check for .tif files
                tif_files_found = True  # At least one file was found
                input_raster_path = os.path.join(root, filename)  # Create full path
                output_raster_path = os.path.join(output_folder_path, f"clipped_{filename}")  # Output name

                # Check raster
                if not check_raster(input_raster_path):
                    continue  # If the raster is not valid, proceed to the next

                # Print paths
                print(f"Input path: {input_raster_path}")
                print(f"Output path: {output_raster_path}")

                # Call the function
                clip_raster_with_shapefile(input_raster_path, shapefile_path, output_raster_path)

    if not tif_files_found:
        print(f"No .tif files found in the folder: {input_folder_path}")