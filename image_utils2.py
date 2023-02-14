# import cv2
# import numpy as np
# import tensorflow as tf
# import pytesseract
#
# # Load the frozen_east_text_detection.pb file
# detection_graph = tf.Graph()
# with detection_graph.as_default():
#     od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile("frozen_east_text_detection.pb", 'rb') as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name='')
#
# def process_image(image_path):
#     # Load the image from disk
#     image = cv2.imread(image_path)
#
#     # Get the image height and width
#     height, width = image.shape[:2]
#
#     # Create a tensor from the image
#     image_tensor = np.expand_dims(image, axis=0)
#
#     # Load the TensorFlow graph
#     with detection_graph.as_default():
#         with tf.Session() as sess:
#             # Get the input and output tensors for the graph
#             image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#             scores = detection_graph.get_tensor_by_name('feature_fusion/Conv_7/Sigmoid:0')
#             geometry = detection_graph.get_tensor_by_name('feature_fusion/concat_3:0')
#
#             # Run inference on the image
#             (scores, geometry) = sess.run([scores, geometry], feed_dict={image_tensor: image_tensor})
#
#     # Apply non-maximum suppression to get the bounding boxes of text regions
#     boxes = cv2.dnn.NMSBoxes(geometry[0], scores[0], score_threshold=0.5, nms_threshold=0.4)
#
#     # Extract the text regions from the image and save as a list
#     text_regions = []
#     for box in boxes:
#         x, y, w, h = geometry[0, box[0], box[1]:box[1]+4] * [width, height, width, height]
#         x1, y1, x2, y2, x3, y3, x4, y4 = cv2.boxPoints([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
#         x1, y1, x2, y2, x3, y3, x4, y4 = np.int0([x1, y1, x2, y2, x3, y3, x4, y4])
#         roi = image[y1:y4, x1:x4]
#         text_regions.append(roi)
#
#     # Extract text from the text regions using OCR and save as a list
#     text_list = []
#     for region in text_regions:
#         text = pytesseract.image_to_string(region)
#         text_list.append(text)
#
#     # Return the combined text from all the regions
#     return '\n'.join(text_list)
import cv2
import numpy as np
import pytesseract
import tensorflow as tf

# Load the frozen_east_text_detection.pb file
od_graph_def = tf.compat.v1.GraphDef()
with tf.compat.v1.gfile.GFile("frozen_east_text_detection.pb", 'rb') as file:
    serialized_graph = file.read()
    od_graph_def.ParseFromString(serialized_graph)

def process_image(image_path):
    # Load the image from disk
    image = cv2.imread(image_path)

    # Get the image height and width
    height, width = image.shape[:2]

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(width, height),
                                 mean=(123.68, 116.78, 103.94), swapRB=True, crop=False)

    # Set the input to the network
    with tf.compat.v1.Session() as sess:
        tf.import_graph_def(od_graph_def, name='')
        inputs = sess.graph.get_tensor_by_name('input_1:0')
        feature_fusion_conv_7 = sess.graph.get_tensor_by_name('feature_fusion/Conv_7/Sigmoid:0')
        feature_fusion_concat_3 = sess.graph.get_tensor_by_name('feature_fusion/concat_3:0')
        scores, geometry = sess.run([feature_fusion_conv_7, feature_fusion_concat_3], feed_dict={inputs: blob})

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
