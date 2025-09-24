import cv2
import pytesseract
import easyocr
import re
import json


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



def extract_aadhaar_hybrid(filepath):
    img = cv2.imread(filepath)

    # --- Step 1: Use Tesseract (fast) for numbers ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tess_config = "--psm 6"
    tess_text = pytesseract.image_to_string(gray, config=tess_config)

    lines = [line.strip() for line in tess_text.split("\n") if line.strip()]  # remove blanks

    # Filter out junk (govt logos, random symbols, etc.)
    cleaned = []
    for line in lines:
        if "GOVERNMENT" in line.upper():  # skip headers
            continue
        if re.match(r'^[\W_]+$', line):  # skip lines with only symbols
            continue
        cleaned.append(line)
    return cleaned

def extract_aadhaar_from_lines(lines):
    details = {}
    name_found = False

    for line in lines:
        clean_line = line.strip()

        # DOB
        dob_match = re.search(r'\d{2}/\d{2}/\d{4}', clean_line)
        if dob_match:
            details["DOB"] = dob_match.group(0)

        # Gender
        if "MALE" in clean_line.upper():
            details["Gender"] = "MALE"
        elif "FEMALE" in clean_line.upper():
            details["Gender"] = "FEMALE"

        # Aadhaar Number
        aadhaar_match = re.search(r'\d{4}\s\d{4}\s\d{4}', clean_line)
        if aadhaar_match:
            details["Aadhaar_Number"] = aadhaar_match.group(0)

        # Name (only take BEFORE DOB/Gender/Number is found)
        if not name_found and re.search(r'[A-Za-z]', clean_line):
            if not any(x in clean_line.upper() for x in ["DOB", "MALE", "FEMALE"]) \
               and not re.search(r'\d{4}\s\d{4}\s\d{4}', clean_line):
                # Clean unwanted characters
                name = re.sub(r'[^A-Za-z\s]', '', clean_line).strip()
                if len(name.split()) >= 2:   # ensure it's at least two words
                    details["Name"] = name
                    name_found = True  # stop after first valid name

    return details




filepath = "aadhaar_img.jpg"
lines = extract_aadhaar_hybrid(filepath)

print(extract_aadhaar_from_lines(lines))
