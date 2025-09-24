import cv2
import easyocr
import re
import json

def extract_pan_details(filepath):
    # Load image
    img = cv2.imread(filepath)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Run OCR
    results = reader.readtext(img)

    extracted_texts = [res[1] for res in results]

    # Join text for regex searching
    full_text = " ".join(extracted_texts)

    details = {}

    # PAN Number (format: 5 letters + 4 digits + 1 letter)
    pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', full_text)
    if pan_match:
        details["PAN_Number"] = pan_match.group(0)

    # Name (first capital words, before Father's Name usually)
    for text in extracted_texts:
        if "Name" in text and "Father" not in text:
            # Next line is usually the Name
            idx = extracted_texts.index(text)
            if idx + 1 < len(extracted_texts):
                details["Name"] = extracted_texts[idx + 1].strip()

    # Father's Name
    for text in extracted_texts:
        if "Father" in text:
            idx = extracted_texts.index(text)
            if idx + 1 < len(extracted_texts):
                details["Father_Name"] = extracted_texts[idx + 1].strip()

    # Date of Birth (format: DD/MM/YYYY)
    dob_match = re.search(r'\d{2}/\d{2}/\d{4}', full_text)
    if dob_match:
        details["Date_of_Birth"] = dob_match.group(0)

    return json.dumps(details, indent=4)



print(extract_pan_details("pan_img.jpg"))
