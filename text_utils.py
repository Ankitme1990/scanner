import pytesseract


def extract_text_from_image(image):
    # Extract the text using OCR
    text = pytesseract.image_to_string(image)
    return text


def write_text_to_file(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
