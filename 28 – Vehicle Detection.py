import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

img = cv2.imread(
    r"C:\Users\DELL\Documents\computer vision lab\vehicle.jpeg"
)

if img is None:
    print("Image not found!")
else:
    results = model(img)
    vehicle_count = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])

            if class_id in [2, 3, 5, 7]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[class_id]

                cv2.rectangle(
                    img, (x1, y1), (x2, y2), (0, 255, 0), 2
                )

                cv2.putText(
                    img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

                vehicle_count += 1

    print("Number of vehicles detected:", vehicle_count)

    cv2.imshow("Vehicle Detection", img)

    cv2.imwrite(
        r"C:\Users\DELL\Documents\computer vision lab\vehicle_detection.png",
        img
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
