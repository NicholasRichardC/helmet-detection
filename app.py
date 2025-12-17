# app.py
import streamlit as st
import cv2
import numpy as np
from detector import detect_helmets

st.set_page_config(
    page_title="Helmet Detection",
    layout="wide"
)

st.title("Helmet Detection System")

conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
             caption="Original Image", use_container_width=True)

    if st.button("🔍 Detect Helmet"):
        with st.spinner("Processing..."):
            results = detect_helmets(image, conf_threshold)

            output = image.copy()

            helmet_count = 0
            no_helmet_count = 0

            for r in results:
                fx, fy, fw, fh = r["face"]
                hx, hy, hw, hh = r["helmet_region"]
                conf = r["confidence"]

                if r["has_helmet"]:
                    color = (0, 255, 0)
                    label = f"Helmet {conf:.2f}"
                    helmet_count += 1
                else:
                    color = (0, 0, 255)
                    label = f"No Helmet {conf:.2f}"
                    no_helmet_count += 1

                cv2.rectangle(output, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
                cv2.rectangle(output, (hx, hy), (hx+hw, hy+hh), color, 2)
                cv2.putText(output, label, (hx, hy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                     caption="Detection Result", use_container_width=True)

            st.success(
                f"Helmet: {helmet_count} | No Helmet: {no_helmet_count}"
            )

