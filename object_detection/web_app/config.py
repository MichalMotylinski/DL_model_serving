"""
This file contains list of classes to be used by web app.
Currently only debugging and production settings are available.
"""


class Config(object):
    """
    Base class with debugging settings turned on
    """
    DEBUG = True
    DEVELOPMENT = True
    SEND_FILE_MAX_AGE_DEFAULT = 0
    UPLOAD_FOLDER = "uploads"
    DETECTION_FOLDER = "detections"
    SITE_IMAGES_FOLDER = "site_images"
    CSS_FOLDER = "css"
    PATH_TO_LABELS = "./data/label_map.pbtxt"
    NUM_CLASSES = 4
    ALLOWED_EXTENSIONS = set([".jpg", ".png", ".jpeg"])


class ProductionConfig(Config):
    """
    Config class intended for production environment with debugging features turned off.
    """
    DEVELOPMENT = False
    DEBUG = False
