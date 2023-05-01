from ultralytics import YOLO
import cv2
import cvzone
import math
cap = cv2.VideoCapture("output_segment_6.mp4")


model = YOLO('weights/yolov8m.pt')

classNames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
   "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
   "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
   "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
   "teddy bear", "hair drier", "toothbrush"]

cod = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('task1_yolo.mp4', cod, 30.0, (480, 848))

while True:
    success, img = cap.read()
    img = cv2.resize(img, (640, 480))

    results = model(img, stream=True, classes=[0, 56])
    classlist = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            w, h = x2 - x1, y2 - y1
            # confidence threshold
            conf = math.ceil((box.conf[0] * 100)) / 100
            # class name
            cls = int(box.cls[0])
            classlist.append(classNames[cls])
            if classNames[cls] == "person":
                cv2.putText(img, f"{classNames[cls]}", (max(0, x1), max(40, y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            elif classNames[cls] == "chair":
                cv2.putText(img, f"{classNames[cls]}", (max(0, x1), max(40, y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    # count chairs and people
    chair_count = classlist.count('chair')
    person_count = classlist.count('person')

    # calculate difference between chairs and people
    chair_diff = chair_count - person_count

    cv2.rectangle(img, (0, 0), (200, 110), (255, 255, 255), -1)
    # display counts on screen
    cv2.putText(img, f"Chairs: {chair_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA, False)
    cv2.putText(img, f"People: {person_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
    cv2.putText(img, f"vacant: {chair_diff}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA, False)

    out.write(img)
    cv2.imshow("imgae", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
