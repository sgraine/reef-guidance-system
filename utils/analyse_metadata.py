import os
from PIL import Image
import piexif
import argparse

def get_altitude(exif_data):
    gps_ifd = exif_data.get("GPS", {})
    altitude = gps_ifd.get(piexif.GPSIFD.GPSAltitude)
    if altitude:
        # Convert from rational to float
        return altitude[0] / altitude[1]
    return None

def find_min_max_altitude(directory):
    min_alt = float('inf')
    max_alt = float('-inf')

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            filepath = os.path.join(directory, filename)
            try:
                img = Image.open(filepath)
                exif_data = piexif.load(img.info.get('exif', b''))
                altitude = get_altitude(exif_data)

                # print(altitude)

                if altitude is not None and altitude < 15:
                    min_alt = min(min_alt, altitude)
                    max_alt = max(max_alt, altitude)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if min_alt == float('inf') or max_alt == float('-inf'):
        print("No valid altitude data found.")
    else:
        print(f"Minimum Altitude: {min_alt} m")
        print(f"Maximum Altitude: {max_alt} m")

if __name__=="__main__":

    # Example usage
    parser = argparse.ArgumentParser(description='Analyze image metadata.')
    parser.add_argument('--directory', type=str, required=True, help='Path to the directory containing images')
    args = parser.parse_args()

    find_min_max_altitude(args.directory)
