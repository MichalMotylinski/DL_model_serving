from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file
from werkzeug.utils import secure_filename
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import time
import cv2
import grpc
from grpc.beta import implementations
import collections
import math
import operator

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2, get_model_metadata_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from utils import label_map_util
from utils import visualization_utils as viz_utils
from core.standard_fields import DetectionResultFields as dt_fields

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

sys.path.append("..")

tf.get_logger().setLevel("ERROR")

PATH_TO_LABELS = "./data/label_map.pbtxt"
NUM_CLASSES = 4

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)

category_index = label_map_util.create_category_index(categories)

app = Flask(__name__)
# Set maximum age of a cashed file
app.config.from_object('config.ProductionConfig')


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]


def get_stub(host="127.0.0.1", port="8500"):
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def load_input_tensor(input_image):
    image_np = load_image_into_numpy_array(input_image)
    image_np_expanded = np.expand_dims(image_np, axis=0).astype(np.uint8)
    tensor = tf.make_tensor_proto(image_np_expanded)
    return tensor


def check_files(filename, folder):
    num = 0
    for f in os.listdir(os.path.join("static", folder)):
        if filename.split(".")[0] in f and filename.split(".")[1] == f[-3:]:
            num = num + 1
    return num


def inference(frame, stub, model_name="od"):
    #stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    stub = get_stub()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "od"

    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_im)
    print(image.size)
    # Resize images to fit the model
    old_width, old_height = image.size
    width = 0
    height = 0
    if old_width != 768:
        width = 768
    if old_height != 1024:
        height = 1024
    image = image.resize((width, height))
    input_tensor = load_input_tensor(image)
    request.inputs["input_tensor"].CopyFrom(input_tensor)

    result = stub.Predict(request, 60.0)

    image_np = load_image_into_numpy_array(image)

    output_dict = {}
    output_dict['detection_classes'] = np.squeeze(
        result.outputs[dt_fields.detection_classes].float_val).astype(np.uint8)
    output_dict['detection_boxes'] = np.reshape(
        result.outputs[dt_fields.detection_boxes].float_val, (-1, 4))
    output_dict['detection_scores'] = np.squeeze(
        result.outputs[dt_fields.detection_scores].float_val)


    threshold = .6
    detections = []
    # Early breaks not to loop through too many elements unnecessarily
    for score in output_dict["detection_scores"]:
        if score > threshold:
            for label in label_map.item:
                index = np.where(output_dict["detection_scores"] == score)[0][0]
                if label.id == output_dict["detection_classes"][index]:
                    detections.append((label.name,
                                       f"{math.floor(score * 100)}%",
                                       output_dict["detection_boxes"][index][0]))
                    break
        else:
            break

    detections = sorted(detections, key=operator.itemgetter(2))

    frame = viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict["detection_boxes"],
        output_dict["detection_classes"],
        output_dict["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=50,
        line_thickness=4,
        min_score_thresh=threshold,
        agnostic_mode=False,
        skip_labels=True)
    return frame, detections, (old_width, old_height)


@app.route("/")
def index():
    stylesheet = url_for("static", filename=os.path.join(app.config["CSS_FOLDER"], "index-style.css"))
    index = url_for("index")
    logo = url_for("static", filename=os.path.join(app.config["SITE_IMAGES_FOLDER"], "bird_logo.jpg"))
    #logo = url_for(app.config["SITE_IMAGES_FOLDER"], filename="bird_logo.jpg")
    return render_template("index.html", stylesheet=stylesheet, index=index, logo=logo)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    num = 0
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if os.path.exists(os.path.join("static", app.config["UPLOAD_FOLDER"], filename)):
            os.rename(os.path.join("static", app.config["UPLOAD_FOLDER"], filename),
                      os.path.join("static", app.config["UPLOAD_FOLDER"],
                                   f"{filename.split('.')[0]}_{check_files(filename, app.config['UPLOAD_FOLDER'])}.{filename.split('.')[1]}"))

        file.save(os.path.join("static", app.config["UPLOAD_FOLDER"], filename))
        return redirect(url_for("results", filename=filename))


@app.route("/results/<filename>")
def results(filename):
    if len(os.listdir("static/" + app.config["DETECTION_FOLDER"])) == len(os.listdir("static/" + app.config["UPLOAD_FOLDER"])):
        return redirect(url_for("index"))
    original = filename
    print(filename)
    detected = filename.rsplit(".")[0] + "_detect." + filename.rsplit(".")[1]
    if os.path.exists(os.path.join("static", app.config["DETECTION_FOLDER"], detected)):
        os.rename(os.path.join("static", app.config["DETECTION_FOLDER"], detected),
                  os.path.join("static", app.config["DETECTION_FOLDER"],
                               f"{detected.split('.')[0]}_{check_files(filename,app.config['DETECTION_FOLDER'])}.{detected.split('.')[1]}"))

    test_image_paths = [os.path.join("static", app.config["UPLOAD_FOLDER"], filename.format(i)) for i in range(1, 2)]

    stub = get_stub()
    new_filename = ""
    for image_path in test_image_paths:
        image_np = np.array(Image.open(image_path))
        image_np_inf, detections, size = inference(image_np, stub)
        im_rgb = image_np_inf[:, :, [2, 1, 0]]
        im = Image.fromarray(im_rgb)
        im = im.resize(size)
        new_filename = filename.rsplit(".", 1)[0] + "_detect." + filename.rsplit(".", 1)[1]
        # im.save(os.path.join("upload/", filename))

        im.save(os.path.join("static", app.config["DETECTION_FOLDER"], new_filename))

    stylesheet = url_for("static", filename=os.path.join(app.config["CSS_FOLDER"], "results-style.css"))
    original_file = url_for("static", filename=os.path.join(app.config["UPLOAD_FOLDER"], original))
    detections_file = url_for("static", filename=os.path.join(app.config["DETECTION_FOLDER"], detected))
    detected_path = os.path.join("static", app.config["DETECTION_FOLDER"], detected)
    index = url_for("index")
    logo = url_for("static", filename=os.path.join(app.config["SITE_IMAGES_FOLDER"], "bird_logo.jpg"))
    print(detected_path)

    return render_template("results.html", stylesheet=stylesheet, original=original_file, detected=detections_file,
                           index=index, logo=logo, result=detections, filename=filename, d_path=detected_path)


@app.after_request
def add_header(response):
    """
    Set up browser cashing policies.
    For object detector no images should be cashed to avoid situation where wrong image is returned.
    This can happen if newly uploaded image has the same name as image uploaded previously.
    """
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post - check = 0, pre - check = 0, max - age = 0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
