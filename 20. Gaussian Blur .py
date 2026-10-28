import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'king.jpg'
try:
    # Load the image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define a 5x5 structuring element (kernel)
    kernel = np.ones((5, 5), np.uint8)

    # Apply Dilation
    dilated_img = cv2.dilate(gray, kernel, iterations=1)

    # Display the results side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title("Original Grayscale")
    axes[0].axis('off')

    axes[1].imshow(dilated_img, cmap='gray')
    axes[1].set_title("Dilated Image (Expanded Foreground)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"An error occurred: {e}. Please ensure '{image_path}' is available.")
