print("Importing packages...")
import onnxruntime as ort
import numpy as np
import cv2
import os
from PIL import Image
from tqdm import tqdm
from PIL.ExifTags import TAGS, GPSTAGS
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import folium

print("Necessary packages imported.")

def img_to_grid(img, row, col):
    ww = [[i.min(), i.max()] for i in np.array_split(range(img.shape[0]),row)]
    hh = [[i.min(), i.max()] for i in np.array_split(range(img.shape[1]),col)]
    grid = [img[j:jj+1,i:ii+1,:] for j,jj in ww for i,ii in hh]
    return grid, len(ww), len(hh)

# Define the cropper function using PyTorch operations
def cropper(images, width, height):
    # Get the current size (B, C, H, W)
    C, H, W = images.shape

    # Calculate the cropping coordinates (center crop or simple top-left crop)
    start_x = (W - width) // 2
    start_y = (H - height) // 2

    # Crop the image (C, H, W) -> (C, height, width)
    cropped_images = images[:, start_y:start_y+height, start_x:start_x+width]

    return cropped_images

def preprocess_image(image, input_shape, im_or_patch:str):

    if im_or_patch=='im':
        # convert to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to float32 for precision, as required for normalization
        image = image.astype(np.float32) / 255.0

        # Normalize the image with ImageNet values (mean and std)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # Reshape to (1, 1, 3) for broadcasting
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)   # Reshape to (1, 1, 3) for broadcasting

        # Perform the normalization
        image = (image - mean) / std

        # resize image
        if image.shape[:2] != input_shape:
            image = cv2.resize(image, input_shape)

        # Change the image from HWC to CWH format (channels first)
        image = np.transpose(image, (2, 1, 0))

        # reshape to (1, 3, width, height)
        reshaped = np.expand_dims(image, axis=0)

        return reshaped.astype(np.float32)

    elif im_or_patch=='patch':

        # convert to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Divide the full image into a grid of patches
        grid, _, _ = img_to_grid(image,4,7)  # row, column

        all_patches = []
        for patch in grid:

            # Change the image from HWC to CWH format (channels first)
            patch = np.transpose(patch, (2, 0, 1)) 

            # Crop to size from grid
            patch_crop = cropper(patch, 758, 760)

            patch = np.transpose(patch_crop, (1, 2, 0))  # Convert (C, W, H) to (H, W, C)

            # Resize to 256,256
            patch = np.array(Image.fromarray(patch).resize((256,256)))

            # Convert the image to float32 for precision, as required for normalization
            # Rescale values to between 0 and 1
            patch = patch.astype(np.float32) / 255.0
            patch = np.transpose(patch, (2, 0, 1))   # Convert (H, W, C) to (C, W, H)

            # Normalize the image with ImageNet values (mean and std)
            mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)  # Reshape to (1, 1, 3) for broadcasting  
            std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)   # Reshape to (1, 1, 3) for broadcasting

            # Perform the normalization
            patch = (patch - mean) / std
                
            all_patches.append(patch)

        reshaped = np.asarray(all_patches)

    return reshaped.astype(np.float32)

# Extract GPS info from image
def get_gps_info(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    gps_info = {}
    if exif_data:
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag)
            if decoded == "GPSInfo":
                for gps_tag, gps_value in value.items():
                    gps_decoded = GPSTAGS.get(gps_tag)
                    gps_info[gps_decoded] = gps_value
    return gps_info

# Convert GPS to latitude and longitude
def get_decimal_from_dms(dms, ref):
    degrees = dms[0] + dms[1] / 60 + dms[2] / 3600
    if ref in ['S', 'W']:
        degrees = -degrees
    return degrees

def get_lat_lon(gps_info):
    lat = get_decimal_from_dms(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
    lon = get_decimal_from_dms(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
    return lat, lon

# Perform inference (replace with your actual model inference code)
def perform_inference(image_path):
    image = cv2.imread(image_path)
    input_np = preprocess_image(image, (256,256), 'patch')

    # Prepare the input in the expected format for ONNX Runtime
    input_name = ort_session.get_inputs()[0].name
    output_names = [output.name for output in ort_session.get_outputs()]

    # Run the ONNX model
    outputs = ort_session.run(output_names, {input_name: input_np})

    # Extract outputs (in the order defined in your model)
    # class_pred = outputs[0]     # Softmax probabilities
    # ratio = outputs[1]          # Predicted class (argmax)
    # deploy = outputs[2]         # Ratio of top two softmax values

    return outputs

if __name__ == '__main__':
    # Load the ONNX model
    onnx_model_path = "outputs/models/onnx/Mobilenet-28-3-256-256.onnx"
    ort_session = ort.InferenceSession(onnx_model_path)

    INPUT_FOLDER = '../CleanData/Evaluation/Deployment/Combined'
    class_names = ['No-Deploy','Deploy']

    # Get list of items in the directory
    contents = os.listdir(INPUT_FOLDER)

    # Separate files and directories
    files = [f for f in contents if os.path.isfile(os.path.join(INPUT_FOLDER, f))]
    folders = [f for f in contents if os.path.isdir(os.path.join(INPUT_FOLDER, f))]

    ################### If you ONLY have a folder with images (no labels) #########################
    # Check if the directory contains images
    if any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')) for f in files):
        print("Executing image processing loop...")

        # List to store results
        results = []

        count = 0
        for image_name in tqdm(files):
            # Perform inference
            decision = perform_inference(os.path.join(INPUT_FOLDER, image_name))

            # Get GPS coordinates
            gps_info = get_gps_info(os.path.join(INPUT_FOLDER, image_name))

            if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                lat, lon = get_lat_lon(gps_info)
                # Append to results list
                results.append({
                    'latitude': lat,
                    'longitude': lon,
                    'deploy_status': decision[2]
                })

        # Convert results to a DataFrame and then a GeoDataFrame
        df = pd.DataFrame(results)
        df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')

        # Create a base map centered at the mean latitude and longitude
        mean_lat = gdf['latitude'].mean()
        mean_lon = gdf['longitude'].mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

        # Add smaller, simpler CircleMarkers to the map
        for idx, row in gdf.iterrows():
            color = 'green' if row['deploy_status'] == 1 else 'red'
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,  # Smaller radius for dense plotting
                color=color,
                fill=True,
                fill_opacity=0.6
            ).add_to(m)

        # Save the map to an HTML file or display it
        m.save("outputs/maps/html_file_inferences.html")
        print("HTML file for model inferences saved.")

    ################### If you have a folder containing subdirectories for Deploy/No-Deploy #########################
    # Check if the directory contains exactly two subfolders - in this case it has been labelled by an ecologist
    elif len(folders) == 2:
        print("Executing folder processing loop...")

        gps_results = []
        count = 0

        for class_num in range(0,2):
            
            class_name = class_names[class_num]
            print(f"Processing folder: ",class_name)
            current_path = os.path.join(INPUT_FOLDER, class_name)

            all_images = [i for i in os.listdir(current_path)]

            print("Processing images...")
            for filename in tqdm(all_images):
                # Perform inference
                decision = perform_inference(os.path.join(current_path, filename))
           
                # Get GPS coordinates
                gps_info = get_gps_info(os.path.join(current_path, filename))

                if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                    lat, lon = get_lat_lon(gps_info)
                    # Append to results list
                    gps_results.append({
                        'latitude': lat,
                        'longitude': lon,
                        'deploy_status': decision[2],
                        'label': class_num
                    })

        ##################### Create Inference Map ########################
        # Convert results to a DataFrame and then a GeoDataFrame
        df = pd.DataFrame(gps_results)
        df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')

        # Create a base map centered at the mean latitude and longitude
        mean_lat = gdf['latitude'].mean()
        mean_lon = gdf['longitude'].mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

        # Add CircleMarkers to the map
        for idx, row in gdf.iterrows():
            color = 'green' if row['deploy_status'] == 1 else 'red'
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,  # Smaller radius for dense plotting
                color=color,
                fill=True,
                fill_opacity=0.6
            ).add_to(m)

        # Save the map to an HTML file or display it
        m.save("outputs/maps/html_file_inferences.html")
        print("HTML file for model inferences saved.")

        ##################### Create Label Map ########################
        # Convert results to a DataFrame and then a GeoDataFrame
        df = pd.DataFrame(gps_results)
        df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')

        # Create a base map centered at the mean latitude and longitude
        mean_lat = gdf['latitude'].mean()
        mean_lon = gdf['longitude'].mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

        # Add CircleMarkers to the map
        for idx, row in gdf.iterrows():
            color = 'green' if row['label'] == 1 else 'red'
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,  # Smaller radius for dense plotting
                color=color,
                fill=True,
                fill_opacity=0.6
            ).add_to(m)

        # Save the map to an HTML file or display it
        m.save("outputs/maps/html_file_ecologist_labels.html")
        print("HTML file for ecologist labels saved.")

    else:
        print("Directory does not match expected configurations.")    

    print("Script end.")