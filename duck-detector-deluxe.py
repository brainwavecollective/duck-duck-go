import cv2
from ultralytics import YOLO
import numpy as np
import time

# Initialize video capture
cap = cv2.VideoCapture(1)

# Initialize both models
duck_model = YOLO('https://huggingface.co/brainwavecollective/yolo8n-rubber-duck-detector/resolve/main/yolov8n_rubberducks4.pt')
standard_model = YOLO('yolov8n.pt')

def get_coordinates(box_coords):
    x1, y1, x2, y2 = box_coords
    center_x = (x1 + x2) / 2
    bottom_y = y2
    return center_x, bottom_y

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def process_frame(frame, model, is_duck_model=True):
    start_time = time.time()
    results = model(frame, conf=0.4)
    
    # Calculate FPS
    fps = 1.0 / (time.time() - start_time)
    
    valid_boxes = []
    for r in results:
        for box in r.boxes:
            class_name = model.names[int(box.cls[0])]
            # For both models, only show teddy bear class
            if class_name == "teddy bear":
                valid_boxes.append({
                    'coords': box.xyxy[0].tolist(),
                    'confidence': float(box.conf[0])
                })
    
    # Filter overlapping boxes
    filtered_boxes = []
    for i, box in enumerate(valid_boxes):
        should_add = True
        for existing_box in filtered_boxes:
            if calculate_iou(box['coords'], existing_box['coords']) > 0.5:
                if box['confidence'] <= existing_box['confidence']:
                    should_add = False
                    break
        if should_add:
            filtered_boxes.append(box)
    
    # Draw boxes and labels
    processed_frame = frame.copy()
    for box in filtered_boxes:
        x1, y1, x2, y2 = map(int, box['coords'])
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Duck ({box['confidence']:.2f})"
        cv2.putText(processed_frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add FPS counter
    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return processed_frame, len(filtered_boxes), fps

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame with both models
    duck_frame, duck_detections, duck_fps = process_frame(frame, duck_model, True)
    standard_frame, std_detections, std_fps = process_frame(frame, standard_model, False)
    
    # Create side-by-side comparison
    height, width = frame.shape[:2]
    canvas = np.zeros((height, width * 2, 3), dtype=np.uint8)
    
    # Place frames side by side
    canvas[:, :width] = duck_frame
    canvas[:, width:] = standard_frame
    
    # Add labels for each model
    cv2.putText(canvas, "Rubber Duck YOLO", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(canvas, f"Detections: {duck_detections}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.putText(canvas, "Standard YOLOv8", (width + 10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(canvas, f"Detections: {std_detections}", (width + 10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the side-by-side comparison
    cv2.imshow('YOLO Comparison', canvas)
    
    # Break loop with 'q'
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()