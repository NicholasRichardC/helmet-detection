# detector.py
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern

# =========================
# CONFIG
# =========================
MODEL_DIR = "models"

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
PROFILE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_profileface.xml"

HELMET_X_OFFSET = -0.45
HELMET_Y_OFFSET = -0.85
HELMET_WIDTH_SCALE = 1.7
HELMET_HEIGHT_SCALE = 1.5

FEATURE_SIZE = (64, 64)

# =========================
# LOAD CASCADES
# =========================
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
profile_cascade = cv2.CascadeClassifier(PROFILE_CASCADE_PATH)

# =========================
# LOAD MODEL (LAZY)
# =========================
_clf = None
_scaler = None

def load_models():
    global _clf, _scaler
    if _clf is None or _scaler is None:
        _clf = joblib.load(f"{MODEL_DIR}/helmet_classifier.pkl")
        _scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    return _clf, _scaler

# =========================
# FACE DETECTION
# =========================
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

    return faces

# =========================
# HELMET REGION
# =========================
def get_helmet_region(face_bbox, img_shape):
    fx, fy, fw, fh = face_bbox
    h, w = img_shape[:2]

    hx = int(fx + fw * HELMET_X_OFFSET)
    hy = int(fy + fh * HELMET_Y_OFFSET)
    hw = int(fw * HELMET_WIDTH_SCALE)
    hh = int(fh * HELMET_HEIGHT_SCALE)

    hx = max(0, hx)
    hy = max(0, hy)
    hw = min(hw, w - hx)
    hh = min(hh, h - hy)

    return hx, hy, hw, hh

# =========================
# FEATURE EXTRACTION
# =========================
def extract_helmet_features(roi):
    roi = cv2.resize(roi, FEATURE_SIZE)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    features = []

    hog_feat = hog(
        gray, orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )
    features.extend(hog_feat)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [16], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-7)
        features.extend(hist)

    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
    features.extend(lbp_hist)

    edges = cv2.Canny(gray, 50, 150)
    features.append(np.sum(edges > 0) / edges.size)

    features.append(np.mean(hsv[:, :, 2]) / 255.0)
    features.append(np.std(hsv[:, :, 1]) / 255.0)

    return np.array(features)

# =========================
# MAIN DETECTION
# =========================
def detect_helmets(image, conf_threshold=0.5):
    clf, scaler = load_models()
    faces = detect_faces(image)
    results = []

    for face in faces:
        hx, hy, hw, hh = get_helmet_region(face, image.shape)
        roi = image[hy:hy+hh, hx:hx+hw]

        if roi.size == 0:
            continue

        feat = extract_helmet_features(roi)
        feat = scaler.transform([feat])

        prob = clf.predict_proba(feat)[0][1]
        has_helmet = prob >= conf_threshold

        results.append({
            "face": face,
            "helmet_region": (hx, hy, hw, hh),
            "has_helmet": has_helmet,
            "confidence": prob
        })

    return results
