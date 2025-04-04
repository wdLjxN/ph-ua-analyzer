import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("pH & UA 生物标志物分析工具")

uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "bmp", "tif"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    st.image(img, caption="原始图像", use_column_width=True)

    def auto_detect_circles(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 50, param1=50, param2=30, minRadius=10, maxRadius=200)
        if circles is not None:
            return np.uint16(np.around(circles[0])).tolist()
        else:
            return None

    def calculate_pH(img, circle):
        x, y, r = circle
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        b, g, _ = cv2.split(img)
        b_mean = np.mean(b[mask == 255])
        g_mean = np.mean(g[mask == 255]) or 1e-9
        return b_mean / g_mean

    def calculate_ua(img, circle):
        x, y, r = circle
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        return np.mean(s[mask == 255])

    circles = auto_detect_circles(img_cv)
    if circles and len(circles) >= 2:
        circles = sorted(circles[:2], key=lambda c: c[0])
        pH = calculate_pH(img_cv, circles[0])
        ua = calculate_ua(img_cv, circles[1])
        st.success(f"分析结果：pH = {pH:.3f}，UA = {ua:.3f}")
    else:
        st.warning("未检测到足够的圆圈，请上传清晰的图像。")
