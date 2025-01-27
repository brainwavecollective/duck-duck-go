import cv2
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture(1)
#model = YOLO('yolov8n.pt')
model = YOLO('https://huggingface.co/brainwavecollective/yolo8n-rubber-duck-detector/resolve/main/yolov8n_rubberducks.pt')

def get_coordinates(box_coords):
    x1, y1, x2, y2 = box_coords
    center_x = (x1 + x2) / 2
    bottom_y = y2
    return center_x, bottom_y

def calculate_iou(box1, box2):
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    results = model(frame, conf=0.11)
    
    # Store all valid boxes
    valid_boxes = []
    for r in results:
        for box in r.boxes:
            class_name = model.names[int(box.cls[0])]
            if class_name in ["teddy bear", "bird", "sports ball"]:
                valid_boxes.append({
                    'coords': box.xyxy[0].tolist(),
                    'confidence': float(box.conf[0])
                })
    
    # Filter overlapping boxes
    filtered_boxes = []
    for i, box in enumerate(valid_boxes):
        should_add = True
        for existing_box in filtered_boxes:
            if calculate_iou(box['coords'], existing_box['coords']) > 0.5:  # IoU threshold
                if box['confidence'] <= existing_box['confidence']:
                    should_add = False
                    break
        if should_add:
            filtered_boxes.append(box)
    
    # Draw and print coordinates for filtered boxes
    for box in filtered_boxes:
        x1, y1, x2, y2 = map(int, box['coords'])
        x, y = get_coordinates(box['coords'])
        print(f"Duck at x: {x:.1f}, y: {y:.1f}")
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Duck ({box['confidence']:.2f})"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    cv2.imshow('Duck Detection', frame)
    key = cv2.waitKey(333)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
