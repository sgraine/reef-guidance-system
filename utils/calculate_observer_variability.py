import os

def get_image_labels(base_path):
    """
    Creates a dictionary of image labels from the folder structure.
    """
    labels = {}
    for label_class in ["Deploy", "No-Deploy"]:
        folder_path = os.path.join(base_path, label_class)
        for image_name in os.listdir(folder_path):
            labels[image_name] = label_class
    return labels

def compare_labels(dict1, dict2):
    # Ensure both dictionaries refer to the same image set
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    
    all_keys = keys1.union(keys2)
    common_keys = keys1.intersection(keys2)
    missing_in_dict1 = keys2 - keys1
    missing_in_dict2 = keys1 - keys2

    mismatches = 0
    for key in common_keys:
        if dict1[key] != dict2[key]:
            mismatches += 1
    
    if missing_in_dict1:
        print(f"Images missing in first label set: {len(missing_in_dict1)}")
    if missing_in_dict2:
        print(f"Images missing in second label set: {len(missing_in_dict2)}")

    return len(all_keys), mismatches

if __name__=="__main__":

    overall_total_images = 0
    overall_mismatches = 0

    # sequences = range(1,6)
    sequences = [3,5]

    for i in sequences:
        # Paths to your datasets - change these as needed - could be random vs biased for one labeller, or across two different labellers for the same set 
        first_labelling_path = "../CleanData/Inter- and Intra- Observer Experiments/Sophie/Random/Sequence_"+str(i)  
        second_labelling_path = "../CleanData/Inter- and Intra- Observer Experiments/Amy/Random/Sequence_"+str(i)

        # Load data
        first_labels = get_image_labels(first_labelling_path)
        second_labels = get_image_labels(second_labelling_path)

        # Compare the labellings
        total_images, mismatches = compare_labels(first_labels, second_labels)

        overall_total_images = overall_total_images + total_images
        overall_mismatches = overall_mismatches + mismatches

        # Calculate inter-observer variation
        agreement = total_images - mismatches
        percentage_agreement = (agreement / total_images) * 100 if total_images > 0 else 0

        # Output the results
        print("**********")
        print("Image sequence: ", str(i))
        print(f"Total images compared: {total_images}")
        print(f"Mismatches: {mismatches}")
        print(f"Agreement: {agreement}")
        print(f"Percentage Agreement: {percentage_agreement:.2f}%")
        # break


    # Calculate inter-observer variation
    overall_agreement = overall_total_images - overall_mismatches
    overall_percentage_agreement = (overall_agreement / overall_total_images) * 100 if overall_total_images > 0 else 0
    print("#############################################")
    print(f"Overall Percentage Agreement: {overall_percentage_agreement:.2f}%")
