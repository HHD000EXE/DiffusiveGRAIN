import pyrealsense2 as rs
import cv2
import os
import time
import numpy as np
from matplotlib import cm

folder_path = '/home/haodi/Documents/ExpertLearning/sand_exp_Nov/formal_dataset'
bag_file_paths = os.listdir(folder_path)

pipeline = rs.pipeline()
config = rs.config()

for bag_file_path in bag_file_paths:
    bag_file_path = os.path.join(folder_path, bag_file_path)

    try:
        config.enable_device_from_file(bag_file_path, repeat_playback=False)
        pipeline.start(config)

        playback = pipeline.get_active_profile().get_device().as_playback()
        playback.set_real_time(False)

        frame_width = 640
        frame_height = 480
        depth_frame_width = 1280
        depth_frame_height = 720
        color_video_writer = cv2.VideoWriter(f'{bag_file_path[:-4]}_color.avi',
                                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                             (frame_width, frame_height))
        depth_video_writer = cv2.VideoWriter(f'{bag_file_path[:-4]}_depth.avi',
                                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                             (depth_frame_width, depth_frame_height))

        while True:
            try:
                frames = pipeline.wait_for_frames(5000)  # Wait up to 5000 ms
            except RuntimeError as e:
                print(f"Timeout occurred: {e}")
                break  # Skip to the next iteration of the loop

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Convert BGR to RGB
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Normalize depth image
            depth_min = 590  # Minimum depth in mm
            depth_max = 670  # Maximum depth in mm
            depth_scaled = np.clip(depth_image, depth_min, depth_max)
            depth_scaled = (depth_scaled - depth_min) / (depth_max - depth_min)

            # Apply Matplotlib 'coolwarm' colormap
            colormap = cm.get_cmap('coolwarm')
            depth_colormap = (colormap(depth_scaled)[:, :, :3] * 255).astype(np.uint8)  # RGB

            # Convert RGB to BGR for OpenCV
            depth_colormap_bgr = cv2.cvtColor(depth_colormap, cv2.COLOR_RGB2BGR)

            # Write frames
            color_video_writer.write(color_image)
            depth_video_writer.write(depth_colormap_bgr)

            # Display images
            cv2.imshow('Depth Image', depth_colormap_bgr)
            cv2.imshow('Color Image', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except RuntimeError as e:
        print(f"RuntimeError: {e}")
    finally:
        pipeline.stop()
        color_video_writer.release()
        depth_video_writer.release()
        cv2.destroyAllWindows()
        time.sleep(1)  # Brief pause to ensure cleanup

# Consider deleting or resetting the pipeline if necessary
del pipeline
