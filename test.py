import cv2
import numpy as np

def detect_robot_with_visualization(input_video_path, output_video_path):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Unable to open the input video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG codec for AVI
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Initial parameters
    threshold_value = 80  # Threshold for binary conversion
    min_area = 10000  # Minimum area for a contour to be considered
    max_aspect_ratio = 3  # Maximum aspect ratio for rectangles
    min_aspect_ratio = 0.33  # Minimum aspect ratio for rectangles

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the contour and filter by area
            approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)

            if len(approx) == 4 and min_area <= area:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)

                if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Robot", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame and threshold image side by side
        combined = np.hstack((cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), frame))
        cv2.imshow("Threshold | Detection", combined)

        # Wait for a key press and update parameters dynamically
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit the program
            break
        elif key == ord('u'):  # Increase threshold
            threshold_value = min(255, threshold_value + 5)
        elif key == ord('d'):  # Decrease threshold
            threshold_value = max(0, threshold_value - 5)
        elif key == ord('a'):  # Increase minimum area
            min_area += 50
        elif key == ord('z'):  # Decrease minimum area
            min_area = max(50, min_area - 50)

        print(f"Threshold: {threshold_value}, Min Area: {min_area}")

        # Write processed frame to output
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved at {output_video_path}")

# Example usage
input_video_path = 'RobotCapture/20240810_174240_color.avi'  # Path to your input video
output_video_path = 'RobotCapture/output_video.avi'  # Path for the output video
detect_robot_with_visualization(input_video_path, output_video_path)
