
import cv2

img = cv2.imread(
    r"C:\Users\DELL\Documents\computer vision lab\images.jpeg")

if img is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

eyes = eye_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(20, 20)
)

for (x, y, w, h) in eyes:
    cv2.rectangle(
        img, (x, y), (x + w, y + h),
        (0, 255, 0), 2
    )

print("Number of eyes detected:", len(eyes))

cv2.imshow("Eye Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
