"""Small local-only OCR demo for Swasthyam Nidan."""

from pathlib import Path

import cv2
import numpy as np
import pytesseract
from flask import Flask, Response, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

# Windows: install Tesseract with its installer, then uncomment and adjust this line.
# Linux/WSL: sudo apt install tesseract-ocr
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Prepare a document photo for OCR."""
    # Grayscale removes distracting colour and gives OCR a simpler signal.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Deskew fixes rotated camera photos that make text lines hard to segment.
    coordinates = np.column_stack(np.where(gray < 240))
    if len(coordinates) > 20:
        angle = cv2.minAreaRect(coordinates)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.2:
            height, width = gray.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(
                gray, matrix, (width, height),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
            )

    # Adaptive threshold handles shadows and uneven lighting across old paper.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    # Median denoise removes specks and compression noise without erasing strokes.
    return cv2.medianBlur(binary, 3)


@app.get("/")
def index() -> Response:
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/style.css")
def stylesheet() -> Response:
    return send_from_directory(BASE_DIR, "style.css")


@app.post("/api/ocr")
def extract_text() -> Response:
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return Response("<p>Couldn't read that image.</p>", status=400, mimetype="text/html")

    try:
        image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("invalid image")
        text = pytesseract.image_to_string(preprocess_image(image), config="--psm 6").strip()
        if not text:
            raise ValueError("empty OCR result")
        escaped = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        return Response(
            "<main><h1>Extracted text</h1><pre>" + escaped + "</pre><p><a href='/'>Try another image</a></p></main>",
            mimetype="text/html",
        )
    except Exception:
        return Response("<p>Couldn't read that image.</p>", status=422, mimetype="text/html")


if __name__ == "__main__":
    app.run(debug=True)