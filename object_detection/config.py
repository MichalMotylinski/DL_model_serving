class Config(object):
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
    DEVELOPMENT = False
    DEBUG = False
    SEND_FILE_MAX_AGE_DEFAULT = 0
    UPLOAD_FOLDER = "uploads"
    DETECTION_FOLDER = "detections"
    SITE_IMAGES_FOLDER = "site_images"
    CSS_FOLDER = "css"
    PATH_TO_LABELS = "./data/label_map.pbtxt"
    NUM_CLASSES = 4
    ALLOWED_EXTENSIONS = set([".jpg", ".png", ".jpeg"])
