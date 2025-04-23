import os
import random
import string
import csv
import shutil
# import piexif
from PIL import Image
from tqdm import tqdm

def resize_images_with_metadata(img_path, output_path, size):
    # Ensure the file is an image (you can add more extensions if needed)
    with Image.open(img_path) as img:
        # Check if the image has EXIF metadata
        exif_data = img.info.get('exif')

        # Resize the image
        resized_img = img.resize(size)

        # Save the resized image, reapplying EXIF metadata if available
        if exif_data:
            resized_img.save(output_path, exif=exif_data)
        else:
            resized_img.save(output_path)

def randomize_images(source_dir, random_dir, mapping_file, size=None):
    """Randomize filenames from class folders and move them to a single folder."""
    os.makedirs(random_dir, exist_ok=True)
    original_to_random = []

    # Traverse class folders
    for class_folder in os.listdir(source_dir):
        print(class_folder)
        class_path = os.path.join(source_dir, class_folder)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                print("old file", file)
                original_path = os.path.join(class_path, file)
                if os.path.isfile(original_path):
                    
                    # Generate a unique random name
                    random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + os.path.splitext(file)[1]
                    print("new file", random_name)
                    while random_name in [entry[1] for entry in original_to_random]:
                        random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + os.path.splitext(file)[1]
        
                    random_path = os.path.join(random_dir, random_name)
                    
                    print(size)
                    if size is None:
                        # Without resizing:
                        shutil.copy2(original_path, random_path)  # Copy to random directory
                    else:
                        # With resizing:
                        resize_images_with_metadata(original_path, random_path, size)

                    original_to_random.append((original_path, random_name))
                break
        break

    # Save the mapping to a CSV file
    with open(mapping_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Original Path', 'Randomized Name'])
        writer.writerows(original_to_random)

    print(f"Images randomised and saved to {random_dir}. Mapping saved to {mapping_file}.")

def randomize_images_no_subfolders(source_dir, random_dir, mapping_file, size=None):
    """Randomize filenames from class folders and move them to a single folder."""
    os.makedirs(random_dir, exist_ok=True)
    original_to_random = []

    # Traverse class folders
    for file in tqdm(os.listdir(source_dir)):
        # print("old file", file)
        original_path = os.path.join(source_dir, file)
        if os.path.isfile(original_path):
            
            # Generate a unique random name
            random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + os.path.splitext(file)[1]
            # print("new file", random_name)
            while random_name in [entry[1] for entry in original_to_random]:
                random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + os.path.splitext(file)[1]

            random_path = os.path.join(random_dir, random_name)
            
            # print(size)
            if size is None:
                # Without resizing:
                shutil.copy2(original_path, random_path)  # Copy to random directory
            else:
                # With resizing:
                resize_images_with_metadata(original_path, random_path, size)

            original_to_random.append((original_path, random_name))
        # break


    # Save the mapping to a CSV file
    with open(mapping_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Original Path', 'Randomized Name'])
        writer.writerows(original_to_random)

    print(f"Images randomised and saved to {random_dir}. Mapping saved to {mapping_file}.")

def recover_images(random_dir, final_dir, mapping_file):
    """Recover original filenames based on the mapping, keeping them in new class folders."""
    with open(mapping_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header

        for original_path, randomized_name in reader:
            randomized_path = os.path.join(random_dir, randomized_name)
            if os.path.isfile(randomized_path):
                # Determine final class folder from original path
                class_folder = os.path.basename(os.path.dirname(original_path))
                final_class_dir = os.path.join(final_dir, class_folder)
                os.makedirs(final_class_dir, exist_ok=True)

                # Rename the file to its original name in the new class folder
                original_name = os.path.basename(original_path)
                final_path = os.path.join(final_class_dir, original_name)
                print("from: ", randomized_path)
                print("to: ", final_path)
                # shutil.copyfile(randomized_path, final_path)
            # break

    print(f"Images recovered and organized into {final_dir}.")

def recover_images_2(randomised_images, final_images, mapping_file):
    # Make sure output folder exists
    os.makedirs(final_images, exist_ok=True)

    # Load mapping: {randomized_name: original_filename}
    mapping = {}
    with open(mapping_file, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_filename = os.path.basename(row["Original Path"])
            randomized_name = row["Randomized Name"]
            mapping[randomized_name] = original_filename

    # Traverse subfolders (Deploy and No-Deploy)
    for label in ["Deploy", "No-Deploy"]:
        input_dir = os.path.join(randomised_images, label)
        output_dir = os.path.join(final_images, label)
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename not in mapping:
                print(f"Skipping {filename}: not found in mapping")
                continue

            original_name = mapping[filename]
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(output_dir, original_name)

            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_path} -> {dst_path}")
            # break

# Example usage:
# Randomize images(source_dir, random_dir, mapping_file, size)

# randomize_images("Whitsundays/Sequence_2_inference", "Whitsundays/Sequence_2-2", "Whitsundays/Sequence_2_mapping.csv")
# randomize_images("Whitsundays/Sequence_3_inference", "Whitsundays/Sequence_3-2", "Whitsundays/Sequence_3_mapping.csv")
# randomize_images("Whitsundays/Sequence_4_inference", "Whitsundays/Sequence_4-2", "Whitsundays/Sequence_4_mapping.csv")
# randomize_images("Whitsundays/Sequence_5_inference", "Whitsundays/Sequence_5-2", "Whitsundays/Sequence_5_mapping.csv")

# randomize_images_no_subfolders("November-HeronIsland-ReefScan/TestSequences/2024-11-25__20241125_010700_Seq01-Heron Island Larvae Pools", "November-HeronIsland-ReefScan/TestSequences-Random/2024-11-25__20241125_010700_Seq01-Heron Island Larvae Pools", "November-HeronIsland-ReefScan/TestSequences-Random/2024-11-25__20241125_010700_Seq01-Heron Island Larvae Pools.csv", size = (2656,1520))

# Recover images after ecologist labeling
sequence = 3
randomised_images = "CleanData/Inter- and Intra- Observer Experiments/Sophie/Random/archive/Sequence_" + str(sequence)
final_images = "CleanData/Inter- and Intra- Observer Experiments/Sophie/Random/Sequence_" + str(sequence)
mapping_file = "CleanData/Inter- and Intra- Observer Experiments/Sequence_"+str(sequence)+"_mapping.csv"
recover_images_2(randomised_images, final_images, mapping_file)
