import cv2
from ultralytics import YOLO, solutions
import math

model = YOLO("yolov8n.pt")
names = model.model.names

cap = cv2.VideoCapture("traffic.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


KNOWN_WIDTH = 0.5 
FOCAL_LENGTH = 800 


line_pts = [(0, 360), (1280, 360)]

speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=names,
    view_img=True,
)

def calculate_distance(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

def process_frame(img):
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        
            box_width = x2 - x1

            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, box_width)
            confidence = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            class_name = names[cls]

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, f"{class_name},Distance: {distance:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False)

    im0 = process_frame(im0)
    im0 = speed_obj.estimate_speed(im0, tracks)
    
    cv2.imshow("Distance and Speed Estimation", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
