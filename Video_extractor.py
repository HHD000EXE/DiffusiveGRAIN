import cv2
import os
import pandas as pd
from matplotlib import cm
import numpy as np

frame_time = []  # frame number that indicate relative time from motor starting
exca_seq_num = []  # sequence number of excavation actions
image_name = []  # extracted image names


depth_mode = 2  # 0 for RGB frame, 1 for \Delta depth frame, 2 for raw depth frame

if depth_mode == 1 or depth_mode == 2:
    # Open the video file
    folder_path = 'RGB-D(depth video)'
    # Open the csv file contains the first frame number that the motor moves
    csv_file_path = 'video starting frame depth.csv'

else:
    # Open the video file
    folder_path = 'RGB-D(color video)'
    # Open the csv file contains the first frame number that the motor moves
    csv_file_path = 'video starting frame color.csv'
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
# Get a list of file names in the folder
file_names = os.listdir(folder_path)

# # Create a blank mask that is the same size as the image
# mask = np.ones((400, 400, 3), dtype="uint8")
# # Mask the vibration of motor
# points = np.array([[60, 400], [60, 180], [220, 180], [220, 270], [160, 270], [160, 400]], dtype=np.int32)
# # Reshape the points in a form required by polylines
# points = points.reshape((-1, 1, 2))
# # Draw the polygon on the mask with white color
# cv2.polylines(mask, [points], isClosed=True, color=(0, 0, 0), thickness=2)
# cv2.fillPoly(mask, [points], color=(0, 0, 0))

for file_name in file_names:
    # Search for the value in the specified column
    matching_row = df[df['video_names'] == file_name]
    try:
        start_frame = matching_row['starting_frame'].iloc[0]
        frame_distance = matching_row['frame_distance'].iloc[0]
        end_frame = matching_row['end_frame'].iloc[0]
    except:
        print(file_name)
        print("File name doesn't matched")
        break


    video_path = folder_path + '/' + file_name
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    # Initialize variables
    frame_count = 0
    frame_record = []
    output_frames = []


    # Read and process frames
    while True:
        # Read the next frame
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Convert to grayscale if the input frame has 3 channels
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Ensure frame is grayscale


        # Define the coordinates for the top-left and bottom-right corners of the ROI
        if depth_mode == 1 or depth_mode == 2:
            x1, y1 = 200, 100  # Bottom-left corner
            x2, y2 = 500, 400  # Top-right corner
        else:
            x1, y1 = 110, 50  # Bottom-left corner
            x2, y2 = 510, 450  # Top-right corner

        # Crop the ROI from the image
        frame = frame[y1:y2, x1:x2]
        # frame = cv2.flip(frame, 0)  # flip the image vertically
        frame = cv2.resize(frame, (400, 400))  # resize the image
        frame_record.append(frame)

        # Process every frame distance
        if (frame_count-start_frame) % frame_distance == 0 and frame_count >= start_frame:
            if depth_mode == 1 and frame_count != start_frame:
                frame_sub = frame.astype(np.float32) - frame_record[frame_count - frame_distance].astype(np.float32)
                output_frames.append(frame_sub)
            else:
                output_frames.append(frame)

        # Increment frame count
        frame_count += 1
        if depth_mode == 1:
            if frame_count >= end_frame:
                break
        else:
            if frame_count >= end_frame - frame_distance:
                break

    # Release the video file
    cap.release()

    # Save the extracted frames to files
    for i, frame in enumerate(output_frames[:]):
        normalized_diff = frame / 255
        # Apply the coolwarm colormap
        colormap = cm.get_cmap('coolwarm')
        frame_colormap = (colormap((normalized_diff + 1) / 2)[:, :, :3] * 255).astype(np.uint8)
        if depth_mode == 0:
            cv2.imwrite(f"train_images(RGB)/{file_name[:-4]}_{i}.png", frame)
        if depth_mode == 1:
            cv2.imwrite(f"train_images(D)/{file_name[:-4]}_{i}.png", frame_colormap * 2)
        if depth_mode == 2:
            cv2.imwrite(f"train_images(RawD)/{file_name[:-4]}_{i}.png", frame)
        image_name.append(f"{file_name[:-4]}_{i}.png")
        frame_time.append(i)
        # exca_seq_num.append(np.floor(i/22.5))