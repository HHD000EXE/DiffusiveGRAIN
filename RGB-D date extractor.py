import pyrealsense2 as rs
import cv2
import os
import time
import numpy as np

folder_path = 'bag_files/'
bag_file_paths = os.listdir(folder_path)

pipeline = rs.pipeline()
config = rs.config()

# Add post-processing filters
spatial_filter = rs.spatial_filter()  # Spatial smoothing filter
spatial_filter.set_option(rs.option.filter_magnitude, 2)  # Smoothing strength
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)  # Smoothing factor
spatial_filter.set_option(rs.option.filter_smooth_delta, 20)  # Delta threshold

temporal_filter = rs.temporal_filter()  # Temporal smoothing filter
temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)  # Smoothing factor
temporal_filter.set_option(rs.option.filter_smooth_delta, 20)  # Delta threshold

for bag_file_path in bag_file_paths:
    bag_file_path = os.path.join(folder_path, bag_file_path)

    try:
        config.enable_device_from_file(bag_file_path, repeat_playback=False)
        pipeline.start(config)

        playback = pipeline.get_active_profile().get_device().as_playback()
        playback.set_real_time(False)

        frame_width = 640
        frame_height = 480
        depth_frame_width = 640 # 1280
        depth_frame_height = 480 # 720
        color_video_writer = cv2.VideoWriter(f'{bag_file_path[:-4]}_color.avi',
                                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                             (frame_width, frame_height))
        depth_video_writer = cv2.VideoWriter(f'{bag_file_path[:-4]}_depth_gray.mp4',
                                             cv2.VideoWriter_fourcc(*'mp4v'), 15,
                                             (depth_frame_width, depth_frame_height), isColor=False)

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

            # Apply filters to the depth frame
            filtered_depth = spatial_filter.process(depth_frame)
            filtered_depth = temporal_filter.process(filtered_depth)

            depth_image = np.asanyarray(filtered_depth.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Convert BGR to RGB
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Normalize depth image to 0–255 for grayscale representation
            depth_min = 590 - 30  # Minimum depth in mm
            depth_max = 670 - 30  # Maximum depth in mm
            depth_scaled = np.clip(depth_image, depth_min, depth_max)
            depth_scaled = ((depth_scaled - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

            # Write frames
            color_video_writer.write(color_image)
            depth_video_writer.write(depth_scaled)

            # Display images
            cv2.imshow('Depth Image (Grayscale)', depth_scaled)
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
