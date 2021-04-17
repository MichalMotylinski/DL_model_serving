# Standard library imports
import cv2
from flask import Flask, render_template, request, redirect, url_for
import grpc
import math
import numpy as np
import operator
import os
from PIL import Image
import socket
import struct
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from werkzeug.utils import secure_filename

# Local imports
from core.standard_fields import DetectionResultFields as dt_fields
from utils import label_map_util
from utils import visualization_utils as viz_utils

# Start Flask application
app = Flask(__name__)
# Set maximum age of a cashed file
app.config.from_object('config.Config')


"""
Functions
"""


def get_label_map():
    """
    Load label map from file.

    Returns:
         label_map: label map object.
    """
    return label_map_util.load_labelmap(app.config["PATH_TO_LABELS"])


def get_category_index(label_map):
    """
    Convert label map object.

    Args:
        label_map: label_map object.

    Returns:
         category_index: list of categories available to the model.
    """
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=app.config["NUM_CLASSES"],
                                                                use_display_name=True)
    return label_map_util.create_category_index(categories)


def get_stub(port="8500"):
    """
    For development purposes use local host IP.

    For preproduction environment:
        - Docker containerisation - get default gateway IP.

    Args:
        port: port number.

    Returns:
         stub: grpc prediction service object.
    """
    if app.config["DEVELOPMENT"]:
        host = "127.0.0.1"
    else:
        host = get_default_gateway()
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub


def get_default_gateway():
    """
    Returns:
         str: IP address string of a default gateway.
    """
    with open("/proc/net/route") as fh:
        for line in fh:
            fields = line.strip().split()
            if fields[1] != '00000000' or not int(fields[3], 16) & 2:
                # If not default route or not RTF_GATEWAY, skip it
                continue
            return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))


def load_image_into_numpy_array(image):
    """
    Convert image array representation.

    Args:
        image: numpy array image representation.

    Returns:
         arr: reshaped numpy array.
    """
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def load_input_tensor(input_image):
    """
    Convert numpy array to input tensor.

    Args:
        input_image: numpy array representation of the image.

    Returns:
        tensor: input tensor object.
    """
    image_np = load_image_into_numpy_array(input_image)
    image_np_expanded = np.expand_dims(image_np, axis=0).astype(np.uint8)
    tensor = tf.make_tensor_proto(image_np_expanded)
    return tensor


def count_files(filename, folder):
    """
    Count number of filename occurrences in the folder.

    Args:
        filename: name of the file to search for.
        folder: folder name that will be searched for a file.

    Returns:
         num: Number of filename occurrences in the folder.
    """
    num = 0
    for f in os.listdir(os.path.join("static", folder)):
        if filename.split(".")[0] in f and filename.split(".")[1] == f.split(".")[1]:
            num = num + 1
    return num


def rename_files(folder, file):
    """
    Rename target file.

    Args:
        folder: folder containing a file.
        file: name of the file.
    """
    file_path = os.path.join("static", folder, file)
    if os.path.exists(file_path):
        os.rename(file_path,
                  os.path.join("static", folder,
                               f"{file.split('.')[0]}_{count_files(file, folder)}.{file.split('.')[1]}"))


def detect(frame, stub, model_name="od"):
    """
    Detect objects on the supplied image.

    Args:
        frame: array representation of the image.
        stub: grpc prediction service object.
        model_name: name of the folder with object detection models.

    Returns:
        new_img: new image with drawn detection boxes.
        detections: list of detected objects.
    """
    predict_request = predict_pb2.PredictRequest()
    predict_request.model_spec.name = model_name

    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_im)

    input_tensor = load_input_tensor(image)
    predict_request.inputs["input_tensor"].CopyFrom(input_tensor)

    result = stub.Predict(predict_request, 60.0)

    image_np = load_image_into_numpy_array(image)

    output_dict = {
        'detection_classes': np.squeeze(result.outputs[dt_fields.detection_classes].float_val).astype(np.uint8),
        'detection_boxes': np.reshape(result.outputs[dt_fields.detection_boxes].float_val, (-1, 4)),
        'detection_scores': np.squeeze(result.outputs[dt_fields.detection_scores].float_val)}

    threshold = .6
    categories = get_category_index(get_label_map())
    detections = []

    # Get detection scores above given threshold together with the name of the class
    for score in output_dict["detection_scores"]:
        if score > threshold:
            idx = np.where(output_dict["detection_scores"] == score)[0][0]
            detections.append((categories[output_dict["detection_classes"][idx]]["name"],
                               f"{math.floor(score * 100)}%",
                               output_dict["detection_boxes"][idx][0]))
        else:
            break

    detections = sorted(detections, key=operator.itemgetter(2))

    """
    Important note about the box drawing settings:
        - Drawing maximum of 100 boxes because it is unlikely that a single image will contain more objects.
          Moreover this will reduce amount of data being created.
    """

    new_img = viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict["detection_boxes"],
        output_dict["detection_classes"],
        output_dict["detection_scores"],
        get_category_index(get_label_map()),
        use_normalized_coordinates=True,
        max_boxes_to_draw=50,
        line_thickness=4,
        min_score_thresh=threshold,
        agnostic_mode=False,
        skip_labels=False)
    return new_img, detections


"""
Routes
"""


@app.route("/")
def index():
    """
    Set variables for the main page and render it.

    Returns:
        render_template: render index.html template.
    """
    # Set variables for webpage
    page_style = url_for("static", filename=os.path.join(app.config["CSS_FOLDER"], "index_style.css"))
    general_style = url_for("static", filename=os.path.join(app.config["CSS_FOLDER"], "site_style.css"))
    index_page = url_for("index")
    about_page = url_for("about")
    logo = url_for("static", filename=os.path.join(app.config["SITE_IMAGES_FOLDER"], "bird_logo.jpg"))
    wall = []
    for i in range(0, 5):
        wall.append(url_for("static", filename=os.path.join(app.config["SITE_IMAGES_FOLDER"], f"wall{i}.jpg")))
    return render_template("index.html", general_style=general_style, page_style=page_style, index=index_page,
                           extensions=list(app.config["ALLOWED_EXTENSIONS"]), logo=logo, wall=wall, about=about_page)


@app.route("/upload", methods=["POST"])
def upload():
    """
    Get file from the main page and pass it to results page.

    Returns:
        redirect: redirect to results.html template.
    """
    file = request.files["file"]
    filename = secure_filename(file.filename)
    rename_files(app.config["UPLOAD_FOLDER"], filename)

    file.save(os.path.join("static", app.config["UPLOAD_FOLDER"], filename))
    return redirect(url_for("results", filename=filename))


@app.route("/results/<filename>")
def results(filename):
    """
    Perform object detection, set variables for the results page and render it.

    Args:
        filename: name of the uploaded file.

    Returns:
        render_template: render results.html template.
    """
    detected = filename.rsplit(".")[0] + "_detect." + filename.rsplit(".")[1]
    rename_files(app.config["DETECTION_FOLDER"], detected)

    stub = get_stub()

    # Detect objects
    image_np = np.array(Image.open(os.path.join("static", app.config["UPLOAD_FOLDER"], filename)))
    image_np_inf, detections = detect(image_np, stub)
    im_rgb = image_np_inf[:, :, [2, 1, 0]]
    im = Image.fromarray(im_rgb)

    # Save file with object detection boxes
    new_filename = filename.rsplit(".", 1)[0] + "_detect." + filename.rsplit(".", 1)[1]
    im.save(os.path.join("static", app.config["DETECTION_FOLDER"], new_filename))

    # Set variables for webpage
    page_style = url_for("static", filename=os.path.join(app.config["CSS_FOLDER"], "results_style.css"))
    general_style = url_for("static", filename=os.path.join(app.config["CSS_FOLDER"], "site_style.css"))
    original_file = url_for("static", filename=os.path.join(app.config["UPLOAD_FOLDER"], filename))
    detections_file = url_for("static", filename=os.path.join(app.config["DETECTION_FOLDER"], detected))
    detected_path = os.path.join("static", app.config["DETECTION_FOLDER"], detected)
    index_page = url_for("index")
    logo = url_for("static", filename=os.path.join(app.config["SITE_IMAGES_FOLDER"], "bird_logo.jpg"))
    birds = [["Erithacus rubecula - European robin", "https://en.wikipedia.org/wiki/European_robin"],
             ["Periparus ater - Coal tit", "https://en.wikipedia.org/wiki/Coal_tit"],
             ["Pica pica - Eurasian magpie", "https://en.wikipedia.org/wiki/Eurasian_magpie"],
             ["Turdus merula - Common blackbird", "https://en.wikipedia.org/wiki/Common_blackbird"]]

    return render_template("results.html", general_style=general_style, page_style=page_style,
                           original=original_file, detected=detections_file, index=index_page,
                           logo=logo, result=detections, filename=filename,
                           d_path=detected_path, birds=birds)


@app.route("/about")
def about():
    """
    Set variables for the about page and render it.

    Returns:
        render_template: render about.html template.
    """
    # Set variables for webpage
    about_style = url_for("static", filename=os.path.join(app.config["CSS_FOLDER"], "about_style.css"))
    general_style = url_for("static", filename=os.path.join(app.config["CSS_FOLDER"], "site_style.css"))
    index_page = url_for("index")
    logo = url_for("static", filename=os.path.join(app.config["SITE_IMAGES_FOLDER"], "bird_logo.jpg"))
    wall = []
    for i in range(0, 5):
        wall.append(url_for("static", filename=os.path.join(app.config["SITE_IMAGES_FOLDER"], f"wall{i}.jpg")))
    return render_template("about.html", general_style=general_style, page_style=about_style, index=index_page,
                           logo=logo, wall=wall)


@app.after_request
def add_header(response):
    """
    Set up browser cashing policies.

    For object detector no images should be cashed to avoid situation where wrong image is returned.

    This can happen if newly uploaded image has the same name as image uploaded previously.

    Args:
        response: response class from request.

    Returns:
        response: new response.
    """
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate,' \
                                        ' post - check = 0, pre - check = 0, max - age = 0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


# Run the script
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
