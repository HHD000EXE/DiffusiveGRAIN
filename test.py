import cv2
import numpy as np

def detect_robot_with_visualization(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Unable to open the input video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    threshold_value = 130
    min_area = 10000
    max_aspect_ratio = 2
    min_aspect_ratio = 0.5

    # HSV color range for purple obstacles
    lower_purple = np.array([110, 20, 2], dtype=np.uint8)
    upper_purple = np.array([180, 255, 255], dtype=np.uint8)

    # Frame counter to decide when to skip
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ------------------------------------------------
        # Skip detection on frames that are not multiples
        # of 3. We just write them out as-is.
        # ------------------------------------------------
        # if frame_count % 5 != 0:
        #     out.write(frame)
        #     cv2.imshow("Skipping Detection", frame)
        #
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord('q'):
        #         break
        #     continue

        # ============= DETECTION LOGIC (Every 3rd Frame) =============
        # Convert the frame to grayscale for thresholding
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert to HSV for color-based obstacle mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
        mask_non_purple = cv2.bitwise_not(mask_purple)

        # Binary threshold
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

        # Exclude purple regions
        thresh_no_purple = cv2.bitwise_and(thresh, thresh, mask=mask_non_purple)

        # Find contours on the masked threshold
        contours, _ = cv2.findContours(thresh_no_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            rot_rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rot_rect

            if w == 0 or h == 0:
                continue

            # Aspect ratio check
            long_side = max(w, h)
            short_side = min(w, h)
            aspect_ratio = long_side / (short_side if short_side != 0 else 1e-5)

            if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                continue

            # Draw contour
            box_points = cv2.boxPoints(rot_rect)
            box_points = np.int0(box_points)
            cv2.drawContours(frame, [box_points], 0, (0, 255, 0), 2)

            # Label the rectangle
            center_x, center_y = int(cx), int(cy)
            cv2.putText(frame, "Robot", (center_x - 30, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # ============= END DETECTION LOGIC =============

        # Display side-by-side: threshold & detection
        display_thresh = cv2.cvtColor(thresh_no_purple, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((display_thresh, frame))
        cv2.imshow("Threshold - No Purple | Detection (Every 3rd Frame)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('u'):
            threshold_value = min(255, threshold_value + 5)
        elif key == ord('d'):
            threshold_value = max(0, threshold_value - 5)
        elif key == ord('a'):
            min_area += 50
        elif key == ord('z'):
            min_area = max(50, min_area - 50)

        print(f"Frame: {frame_count}, Threshold: {threshold_value}, Min Area: {min_area}")

        # Write the processed frame
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved at {output_video_path}")


# Example usage
input_video_path = 'RobotCapture/demo1.avi'
output_video_path = 'RobotCapture/demo1post.avi'
detect_robot_with_visualization(input_video_path, output_video_path)
