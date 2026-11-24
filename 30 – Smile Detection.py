

import cv2

img = cv2.imread(
    r"C:\Users\DELL\Documents\computer vision lab\images.jpeg"
)

if img is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

if smile_cascade.empty():
    print("Smile cascade file not found")
    exit()

smiles = smile_cascade.detectMultiScale(
    gray,
    scaleFactor=1.7,
    minNeighbors=20,
    minSize=(25, 25)
)

for (x, y, w, h) in smiles:
    cv2.rectangle(
        img, (x, y), (x + w, y + h),
        (0, 255, 0), 2
    )

print("Number of smiles detected:", len(smiles))

cv2.imshow("Smile Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
