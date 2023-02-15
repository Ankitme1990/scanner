import cv2
import numpy as np
import pytesseract

# Load the frozen_east_text_detection.pb file
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

def process_image(image_path):
    # Load the image from disk
    image = cv2.imread(image_path)

    # Get the image height and width
    height, width = image.shape[:2]

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(width, height),
                                 mean=(123.68, 116.78, 103.94), swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Forward pass through the network to get the output
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    # Apply non-maximum suppression to get the bounding boxes of text regions
    boxes = cv2.dnn.NMSBoxes(geometry[0], scores[0], score_threshold=0.5, nms_threshold=0.4)

    # Extract the text regions from the image and save as a list
    text_regions = []
    for box in boxes:
        x, y, w, h = geometry[0, box[0], box[1]:box[1]+4] * [width, height, width, height]
        x1, y1, x2, y2, x3, y3, x4, y4 = cv2.boxPoints([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
        x1, y1, x2, y2, x3, y3, x4, y4 = np.int0([x1, y1, x2, y2, x3, y3, x4, y4])
        roi = image[y1:y4, x1:x4]
        text_regions.append(roi)

    # Extract text from the text regions using OCR and save as a list
    text_list = []
    for region in text_regions:
        text = pytesseract.image_to_string(region)
        text_list.append(text)

    # Return the combined text from all the regions
    return '\n'.join(text_list)
