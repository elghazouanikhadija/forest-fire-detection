import cv2
import os
from ultralytics import YOLO
from datetime import datetime

# Create results directory if it doesn't exist
os.makedirs("results/detected_images", exist_ok=True)

# Path to the video
video_path = "samples/test_video.mp4"

# Load YOLOv8 model
model = YOLO("models/best_fire_detection.pt")

# Load the video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video.")
    exit()

# Counters
frame_count = 0
saved_images_count = 0

print("Starting detection with image saving...")

# Frame processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_count += 1
    has_detection = False

    # YOLOv8 detection
    results = model(frame)

    # Process results
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()       # Bounding box coordinates
        scores = r.boxes.conf.cpu().numpy()      # Confidence scores
        classes = r.boxes.cls.cpu().numpy().astype(int)  # Predicted classes

        for box, score, cls in zip(boxes, scores, classes):
            has_detection = True
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[cls]} {score:.2f}"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # SAVE IMAGE IF DETECTION FOUND
    if has_detection:
        saved_images_count += 1
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        filename = f"results/detected_images/detection_frame_{frame_count:06d}_{timestamp}.jpg"
        
        # Save the image
        cv2.imwrite(filename, frame)
        print(f"✅ Detection saved: {filename}")

    # Display the frame
    try:
        # Add counters to the screen
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Saved images: {saved_images_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("YOLOv8 Detection on Video", frame)
    except Exception as e:
        print("Error displaying with cv2.imshow(). Make sure OpenCV is installed with GUI support.")
        print("Error details:", e)
        break

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Final report
print(f"\n=== PROCESSING COMPLETED ===")
print(f"Frames analyzed: {frame_count}")
print(f"Images with detections saved: {saved_images_count}")
print(f"Image folder: results/detected_images/")