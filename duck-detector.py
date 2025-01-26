import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')

def get_duck_position(box_coords, height=480, width=640):
    far = height/3
    mid = 2*height/3
    left = width/3
    center = 2*width/3
    
    x = (box_coords[0] + box_coords[2])/2
    y = box_coords[3]
    
    distance = "far" if y < far else "mid" if y < mid else "near"
    position = "left" if x < left else "center" if x < center else "right"
    
    return f"{distance}_{position}"

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    results = model(frame, conf=0.3)  # Lowered confidence threshold
    
    for r in results:
        for box in r.boxes:
            # Print all detected objects and their confidence
            class_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            print(f"Detected: {class_name} ({confidence:.2f})")
            
            if class_name == "teddy bear":
                pos = get_duck_position(box.xyxy[0].tolist())
                print(f"Duck at: {pos}")
                
    cv2.imshow('Duck Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()