import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Load the pre-trained model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")

# Load COCO labels for class names
LABELS_FILE = 'coco_labels.txt'
with open(LABELS_FILE, 'r') as f:
    class_names = f.read().splitlines()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor
    input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)

    # Perform object detection
    detections = model(input_tensor)

    # Extract detection results
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    # Draw bounding boxes and labels
    height, width, _ = frame.shape
    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5:  # Confidence threshold
            box = detection_boxes[i] * [height, width, height, width]
            class_id = detection_classes[i]
            label = class_names[class_id - 1]  # Adjust for 0-based index in classes
            confidence = detection_scores[i]

            # Draw bounding box
            y_min, x_min, y_max, x_max = box.astype(int)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
