import os

# Define the paths to the two folders
folder2 = "train_images(RawD)"
folder1 = "train_images(D)"

# Get the list of filenames in both folders
files_in_folder1 = set(os.listdir(folder1))
files_in_folder2 = set(os.listdir(folder2))

# Find the files that are in folder1 but not in folder2
unique_to_folder1 = files_in_folder1 - files_in_folder2

# Print the filenames that are unique to folder1
print("Files in folder1 but not in folder2:")
for filename in unique_to_folder1:
    print(filename)