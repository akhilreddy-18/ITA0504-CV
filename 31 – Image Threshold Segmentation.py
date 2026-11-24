import cv2

# Read the input image
img = cv2.imread(r"C:\Users\DELL\Documents\computer vision lab\images.jpeg")

if img is None:
    print("Error: Image not found!")
else:
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    ret, thresh = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY
    )

    # Display the original and segmented images
    cv2.imshow("Original Image", img)
    cv2.imshow("Grayscale Image", gray)
    cv2.imshow("Threshold Segmentation", thresh)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
