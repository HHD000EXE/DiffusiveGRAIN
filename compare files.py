import os

def compare_folders(folder1, folder2):
    """
    Compare files in two folders and output the file names that exist in one folder but not in the other.

    Parameters:
        folder1 (str): Path to the first folder.
        folder2 (str): Path to the second folder.

    Returns:
        dict: A dictionary with the comparison results.
    """
    files_in_folder1 = set(os.listdir(folder1))
    files_in_folder2 = set(os.listdir(folder2))

    only_in_folder1 = files_in_folder1 - files_in_folder2
    only_in_folder2 = files_in_folder2 - files_in_folder1

    return {
        "only_in_folder1": sorted(only_in_folder1),
        "only_in_folder2": sorted(only_in_folder2),
    }

# Example usage
if __name__ == "__main__":
    folder1 = "train_images(D)"
    folder2 = "train_images(RawD)"

    if not os.path.isdir(folder1) or not os.path.isdir(folder2):
        print("Both paths must be valid directories.")
    else:
        results = compare_folders(folder1, folder2)

        print("\nFiles only in folder 1:")
        for file in results["only_in_folder1"]:
            print(file)

        print("\nFiles only in folder 2:")
        for file in results["only_in_folder2"]:
            print(file)