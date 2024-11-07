import cv2
import numpy as np
from fer import FER

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Initialize FER for mood detection
mood_detector = FER()

while True:
    # Read a frame from the webcam
    ret, image = cap.read()
    if not ret:
        break

    height, width, channels = image.shape

    # Prepare the image for the YOLO model
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get the output from the YOLO model
    outs = net.forward(output_layers)

    # Initialize lists for YOLO detection data
    class_ids = []
    confidences = []
    boxes = []

    # Iterate through each output from YOLO
    for out in outs:
        for detection in out:
            scores = detection[5:]  # get the scores for each class
            class_id = np.argmax(scores)  # get the index of the max score
            confidence = scores[class_id]  # get the max score

            if confidence > 0.5:  # Filter out weak detections
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes for object detection
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Mood detection using FER
    faces = mood_detector.detect_emotions(image)
    for face in faces:
        # Draw bounding box around face
        x, y, w, h = face['box']
        emotions = face['emotions']
        mood = max(emotions, key=emotions.get)  # Get the highest scoring mood
        mood_confidence = emotions[mood]

        # Draw the face bounding box and mood label
        color = (255, 0, 0)  # Blue color for face bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"Mood: {mood} ({mood_confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,
                    2)

    # Display the resulting frame
    cv2.imshow("Real-Time Object and Mood Detection", image)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()