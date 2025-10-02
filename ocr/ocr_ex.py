import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import cv2

from pdf2image import convert_from_path
import os

def extract_rawtxt(filepath):
    text = ""

    # If PDF → convert first page to image
    if filepath.lower().endswith(".pdf"):
        pages = convert_from_path(filepath, dpi=300, first_page=1, last_page=1, poppler_path=r"C:\poppler\poppler-25.07.0\Library\bin")
        page = pages[0]
        temp_img = filepath.replace(".pdf", ".jpg")
        page.save(temp_img, "JPEG")
        img = cv2.imread(temp_img)
        text = pytesseract.image_to_string(img,  config='--psm 6')

        # optional cleanup: remove temp image
        os.remove(temp_img)

    # If JPG/PNG → read directly
    elif filepath.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(filepath)
        text = pytesseract.image_to_string(img,  config='--psm 6')

    else:
        raise ValueError("Unsupported file type")

    return text


def extract_pantxt(filepath):

    text = ""

    # If PDF → convert first page to image
    if filepath.lower().endswith(".pdf"):
        pages = convert_from_path(
            filepath,
            dpi=300,
            first_page=1,
            last_page=1,
            poppler_path=r"C:\poppler\poppler-25.07.0\Library\bin"
        )
        page = pages[0]
        temp_img = filepath.replace(".pdf", ".jpg")
        page.save(temp_img, "JPEG")
        img = cv2.imread(temp_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--psm 6', lang="eng")
        os.remove(temp_img)  # cleanup temporary image

    # If JPG/PNG → read directly
    elif filepath.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--psm 6', lang="eng")

    else:
        raise ValueError("Unsupported file type")

    return text


