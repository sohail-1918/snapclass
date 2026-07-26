import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students


# -----------------------------
# Load dlib models
# -----------------------------
@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(face_recognition_models.pose_predictor_model_location())
    facerec = dlib.face_recognition_model_v1(face_recognition_models.face_recognition_model_location())
    return detector, sp, facerec


# -----------------------------
# Normalize embeddings
# -----------------------------
def normalize_embedding(embedding):
    emb = np.array(embedding, dtype=np.float64)
    return emb / np.linalg.norm(emb)


# -----------------------------
# Get embeddings for ALL faces in one photo
# -----------------------------
def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()
    faces = detector(image_np, 1)

    encodings = []
    for face in faces:
        shape = sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1)
        encodings.append(normalize_embedding(face_descriptor))

    return encodings


# -----------------------------
# Train classifier
# -----------------------------
@st.cache_resource
def get_trained_model():
    X, y = [], []
    student_db = get_all_students()

    if not student_db:
        return None

    for student in student_db:
        embedding = student.get("face_embedding")
        if embedding:
            X.append(normalize_embedding(embedding))
            y.append(student.get("student_id"))

    if len(X) == 0:
        return None

    clf = SVC(kernel="linear", probability=True, class_weight="balanced")
    try:
        clf.fit(X, y)
    except ValueError:
        return None

    return {"clf": clf, "X": X, "y": y}


def train_classifier():
    st.cache_resource.clear()
    model_data = get_trained_model()
    return bool(model_data)


# -----------------------------
# Predict attendance for multiple people
# -----------------------------
# NOTE: threshold is calibrated for L2-NORMALIZED embeddings (unit vectors).
# normalize_embedding() divides each descriptor by its own norm (~0.45-0.5
# for raw dlib descriptors), which roughly doubles the effective distance
# between any two embeddings compared to dlib's raw (non-normalized) output.
# The commonly cited dlib/face_recognition threshold of ~0.6 assumes RAW
# descriptors -- using that number here on normalized embeddings incorrectly
# rejects genuine matches. 1.0 is a starting point; tune based on real
# distance values logged during testing (temporarily re-enable the debug
# st.write below to see actual scores for known-correct matches).
def predict_attendance(class_image_np, threshold=1.0):
    encodings = get_face_embeddings(class_image_np)
    detected_students = {}

    model_data = get_trained_model()
    if not model_data:
        return detected_students, [], len(encodings)

    X_train = model_data["X"]
    y_train = model_data["y"]
    all_students = sorted(list(set(y_train)))

    for encoding in encodings:
        # Compare this face against ALL stored embeddings
        distances = [np.linalg.norm(train_emb - encoding) for train_emb in X_train]
        min_index = int(np.argmin(distances))
        best_match_id = int(y_train[min_index])
        best_match_score = distances[min_index]

        if best_match_score <= threshold:
            detected_students[best_match_id] = True

    return detected_students, all_students, len(encodings)


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.title("Face Recognition Attendance System")

    img_file = st.camera_input("Take a group photo")

    if img_file is not None:
        from PIL import Image

        image = Image.open(img_file)
        image_np = np.array(image.convert("RGB"), dtype=np.uint8)

        detected_students, all_students, num_faces = predict_attendance(image_np)

        if num_faces == 0:
            st.warning("No faces found!")
        else:
            if detected_students:
                st.success(f"Detected students: {list(detected_students.keys())}")
            else:
                st.error("Faces detected but none matched")


if __name__ == "__main__":
    main()