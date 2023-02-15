import cv2
import numpy as np

def process_image(image_path):
    # Load the image from disk
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform Gaussian smoothing to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform image sharpening to enhance the details
    sharp = cv2.filter2D(blur, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

    # Perform thresholding on the image
    threshold_value, img_threshold = cv2.threshold(sharp, 150, 255, cv2.THRESH_BINARY)

    # Convert the thresholded image back to PIL image format
    return img_threshold
