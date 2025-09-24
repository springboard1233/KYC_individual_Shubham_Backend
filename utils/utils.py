import hashlib
import re
import os
import cv2
import pytesseract
from PIL import Image
from pymongo import MongoClient

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- Aadhaar Verhoeff ----------------
d_table = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8],
    [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],
    [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0]
]
p_table = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0],
    [4,2,8,6,5,7,9,3,0,1],
    [2,7,9,3,8,0,6,4,1,5],
    [7,0,4,6,9,1,3,2,5,8]
]
def verhoeff_check(num: str) -> bool:
    c = 0
    num = num[::-1]
    for i, item in enumerate(num):
        c = d_table[c][p_table[(i % 8)][int(item)]]
    if c == 0:
        return True #"aadhaar number is valid"
    else:
        return False #"aadhaar number is not valid"

# ---------------- PAN Regex ----------------
def validate_pan(pan: str) -> bool:
    """PAN format: 5 letters + 4 digits + 1 letter"""
    return bool(re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', pan))

# ---------------- Hashing ----------------
def hash_value(number: str) -> str:
    return hashlib.sha256(number.encode()).hexdigest()

# ---------------- Manipulation Check ----------------
def check_document(filepath):
    issues = []
    valid_exts = [".jpg", ".jpeg", ".png", ".pdf"]
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in valid_exts:
        issues.append("Suspicious file extension")
    try:
        img = Image.open(filepath)
        if img.format.lower() not in ["jpeg", "png", "pdf"]:
            issues.append("Mismatch between extension and real format")
    except:
        issues.append("Unreadable image format")

    size_kb = os.path.getsize(filepath) / 1024
    if size_kb < 15:
        issues.append("File size too small (<15KB)")
    elif size_kb > 5000:
        issues.append("File size too large (>5MB)")

    img_cv = cv2.imread(filepath)
    if img_cv is not None:
        h, w = img_cv.shape[:2]
        aspect_ratio = w / h
        if aspect_ratio < 0.8 or aspect_ratio > 2.0:
            issues.append(f"Unusual aspect ratio ({aspect_ratio:.2f})")
        try:
            ocr_data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in ocr_data["conf"] if conf != "-1"]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            if avg_conf < 50:
                issues.append("Alter_Information , Fake_Text_Layers , Inconsistent_Font") #Low OCR confidence (<50%)
        except:
            issues.append("OCR check failed")

    return (len(issues) == 0), issues

# ---------------- Fraud Score ----------------
def compute_fraud_score(aadhaar, pan):
    score = 0
    #issues = []
    if aadhaar:
        score += aadhaar.get("fraud_score", 0)
        #issues.extend(aadhaar.get("issues", []))
    if pan:
        score += pan.get("fraud_score", 0)
        #issues.extend(pan.get("issues", []))

    score = (score // 2)

    if score < 20:
        risk = "Low"
    elif score < 50:
        risk = "Medium"
    else:
        risk = "High"

    return score, risk#, issues
