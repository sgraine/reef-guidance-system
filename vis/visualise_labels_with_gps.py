print("Importing packages...")
import os
from PIL import Image
from tqdm import tqdm
from PIL.ExifTags import TAGS, GPSTAGS
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import folium

print("Necessary packages imported.")

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

if __name__ == '__main__':
    ecologist = 'Sophie'

    start_path = '../CleanData/Inter- and Intra- Observer Experiments/'+ecologist+'/Random'
    class_names = ['No-Deploy','Deploy']

    # Determine the number of sequences:
    sequence_list = os.listdir(start_path)
    print(sequence_list)
    
    gps_results = []
    for sequence in sequence_list:

        INPUT_FOLDER = os.path.join(start_path, sequence)

        # Get list of items in the directory
        contents = os.listdir(INPUT_FOLDER)

        # Separate files and directories
        files = [f for f in contents if os.path.isfile(os.path.join(INPUT_FOLDER, f))]

        for class_num in range(0,2):
            
            class_name = class_names[class_num]
            print(f"Processing folder: ",class_name)
            current_path = os.path.join(INPUT_FOLDER, class_name)

            all_images = [i for i in os.listdir(current_path)]

            print("Processing images...")
            for filename in tqdm(all_images):       
                # Get GPS coordinates
                gps_info = get_gps_info(os.path.join(current_path, filename))

                if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                    lat, lon = get_lat_lon(gps_info)
                    # Append to results list
                    gps_results.append({
                        'latitude': lat,
                        'longitude': lon,
                        'label': class_num
                    })

    ##################### Create  Map ########################
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
    m.save("outputs/maps/"+ecologist+"_inferences.html")
    print("HTML file for ecologist labels saved.") 

    print("Script end.")