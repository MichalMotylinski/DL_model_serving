class Config(object):
    DEBUG = True
    DEVELOPMENT = True
    SECRET_KEY = 'do-i-really-need-this'
    FLASK_HTPASSWD_PATH = '/secret/.htpasswd'
    FLASK_SECRET = SECRET_KEY
    DB_HOST = 'database' # a docker link


class ProductionConfig(Config):
    DEVELOPMENT = False
    DEBUG = False
    SEND_FILE_MAX_AGE_DEFAULT = 0
    UPLOAD_FOLDER = "uploads"
    DETECTION_FOLDER = "detections"
    SITE_IMAGES_FOLDER = "site_images"
    CSS_FOLDER = "css"
    ALLOWED_EXTENSIONS = set(["jpg", "png", "jpeg"])
    EXT_ERROR = False
