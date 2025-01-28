import cv2
import pytesseract
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# Configure Tesseract path (modify as per your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def preprocess_invoice(image_path):
    """
    Preprocess the invoice image for OCR.
    
    Args:
    - image_path (str): Path to the invoice image.
    
    Returns:
    - ndarray: Processed image ready for OCR.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(binary, 3)
    return denoised

def extract_invoice_text(image_path):
    """
    Extract text from the preprocessed invoice image using Tesseract OCR.
    
    Args:
    - image_path (str): Path to the invoice image.
    
    Returns:
    - str: Extracted text.
    """
    img = preprocess_invoice(image_path)
    text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
    return text

def extract_field(text, pattern):
    """
    Extract a specific field from text using a regex pattern.
    
    Args:
    - text (str): Text to search.
    - pattern (str): Regex pattern to match the field.
    
    Returns:
    - str or None: Matched field or None if no match is found.
    """
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None

def process_invoices(file):
    """
    Process the uploaded invoice file and extract relevant data.
    
    Args:
    - file: Uploaded file (streamlit file_uploader object).
    
    Returns:
    - dict: Extracted data in JSON format.
    - figure: Visualization figure for displaying extracted text.
    """
    temp_path = "temp_invoice.png"  # Save file temporarily
    with open(temp_path, "wb") as f:
        f.write(file.read())
    
    try:
        # Extract text using OCR
        extracted_text = extract_invoice_text(temp_path)
        
        # Parse fields using regex
        invoice_data = {
            "InvoiceNumber": extract_field(extracted_text, r"Invoice\s*#?:?\s*(\d+)"),
            "Date": extract_field(extracted_text, r"Date\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})"),
            "TotalAmount": extract_field(extracted_text, r"Total\s*[:\-]?\s*\$?([\d,]+\.\d{2})"),
            "ExtractedText": extracted_text
        }
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, extracted_text[:500], wrap=True, ha="center", va="center", fontsize=12)
        ax.axis("off")
        
        return invoice_data, fig  # Return data and figure
    
    except Exception as e:
        raise ValueError(f"Error processing invoice: {e}")
