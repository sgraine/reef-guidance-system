import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.cm as cm

import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
import imgaug.augmenters as iaa

from PIL.ExifTags import TAGS, GPSTAGS
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import folium

print("Imports done")

def get_color(value):
    cmap = cm.get_cmap("spring")  # Use the "spring" colormap
    norm_value = max(0, min(1, value))  # Ensure value is between 0 and 1
    return mcolors.to_hex(cmap(norm_value))

# Extract GPS info from image
def get_gps_info(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    gps_info = {}
    if exif_data:
        for tag, value in exif_data.items():
            # print(tag, value)
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

### Data Transformation Functions ###
def transform_func(image):
    'Transform into a pytorch Tensor'
    transform_list = []
    # transform_list.append(transforms.Resize((456,456), interpolation=transforms.InterpolationMode.BICUBIC))  # this is for full size (original) model
    # transform_list.append(transforms.CenterCrop(456))
    transform_list.append(transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC))  # this is for full size (original) model
    transform_list.append(transforms.CenterCrop(256))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])) # imagenet
    transform = transforms.Compose(transform_list)

    return transform(image)#.float()

### Helper Functions ###
class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def img_to_grid(img, row,col):
    ww = [[i.min(), i.max()] for i in np.array_split(range(img.shape[0]),row)]
    hh = [[i.min(), i.max()] for i in np.array_split(range(img.shape[1]),col)]
    grid = [img[j:jj+1,i:ii+1,:] for j,jj in ww for i,ii in hh]
    return grid, len(ww), len(hh)

def cropper(images, width, height):

    seq = iaa.Sequential([
        iaa.CropToFixedSize(width=width, height=height)
    ])

    return seq.augment_image(images)

if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = "models/model-1745301988CKPT.pt"

    model_load = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")

    # Get the number of features in the last layer
    num_ftrs = model_load.classifier[3].in_features

    # Replace the last layer (classifier) with a new one for 4 classes
    model_load.classifier[3] = nn.Sequential(nn.Linear(num_ftrs, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.15),
                                 nn.Linear(512, 3))

    model_load.load_state_dict(torch.load(model_path))

    model_load = model_load.to(device)
    model_load.eval()

    # Class names (folders) in your dataset
    class_names = ['No-Deploy','Deploy']  # Replace with your actual class names

    input_folder = '../CleanData/Evaluation/Deployment/Combined'

    label_dict = {}
    inference_dict = {} 
    gps_results = []

    i = 0

    for class_num in range(0,2):

        class_name = class_names[class_num]

        current_path = os.path.join(input_folder, class_name)

        all_images = [i for i in os.listdir(current_path)]

        print("Processing images...")
        for filename in tqdm(all_images):

            image = Image.open(os.path.join(current_path, filename))
            
            vis_image = Image.open(os.path.join(current_path, filename))
            vis_image = np.array(vis_image)

            row = 4
            col = 7

            width = int(vis_image.shape[1])
            height = int(vis_image.shape[0])

            # Divide the full image into a grid of patches
            grid, _, _ = img_to_grid(vis_image,row,col)

            all_patches = []
            for patch in grid:
                patch_crop = cropper(patch, int(np.floor(width / col)), int(np.floor(height / row)))
                all_patches.append(torch.unsqueeze(transform_func(Image.fromarray(patch_crop)), dim=0))

            all_patches_torch = torch.cat(all_patches, dim=0).to(device)

            soft = torch.nn.Softmax(dim=0)

            outputs_list = []

            with torch.no_grad(): 
                outputs_batch = model_load(all_patches_torch)
                soft_batch = torch.nn.Softmax(dim=1)
                outputs_batch_soft =  soft_batch(outputs_batch)
                outputs_batch_preds = torch.argmax(outputs_batch_soft, dim=1).int()
            

            # Compute ratio
            zero_count = (outputs_batch_preds == 1).sum(dim=0) # 1 for coral patches
            ratio = zero_count / outputs_batch_preds.shape[0]
            ratio = ratio.cpu().numpy()

            # Get GPS coordinates
            gps_info = get_gps_info(os.path.join(current_path, filename))
            # print(gps_info)
            if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                lat, lon = get_lat_lon(gps_info)
                # Append to results list
                gps_results.append({
                    'latitude': lat,
                    'longitude': lon,
                    'deploy_status': ratio,
                })

    ##################### Create Coral Map ########################
    # Convert results to a DataFrame and then a GeoDataFrame
    df = pd.DataFrame(gps_results)
    df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Create a base map centered at the mean latitude and longitude
    mean_lat = gdf['latitude'].mean()
    mean_lon = gdf['longitude'].mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

    # Add CircleMarkers with gradient colors
    for idx, row in gdf.iterrows():
        color = get_color(row['deploy_status'])  # Compute gradient color
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

    # Save the map to an HTML file or display it
    m.save("outputs/maps/coral-map.html")