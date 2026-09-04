# Swasthyam Nidan OCR Demo

Local-only SIH intern demo for extracting text from prescription and report photos.

## Setup

1. Create and activate a virtual environment: `python -m venv .venv`, then `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/WSL.
2. Install Python dependencies: `pip install -r requirements.txt`.
3. Install the Tesseract OCR binary:
	- Windows: install the Tesseract Windows installer, then add its install directory to PATH. If it is not on PATH, uncomment the Windows `tesseract_cmd` line in `app.py`.
	- Linux/WSL: `sudo apt update && sudo apt install tesseract-ocr`.
4. Start the app: `flask --app app run`.
5. Open `http://127.0.0.1:5000`.

The app does not store uploads or require internet access while running. OpenCV preprocesses every image with grayscale, deskew, adaptive thresholding, and denoising before Tesseract.