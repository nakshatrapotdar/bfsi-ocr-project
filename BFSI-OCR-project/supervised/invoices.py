import cv2
import pytesseract
import numpy as np
import pandas as pd
import re
import os
import logging
from PIL import Image
import platform

# Configure Tesseract path dynamically
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_invoice(image_path):
    """
    Preprocess the invoice image for OCR.
    Converts to grayscale, applies thresholding and noise removal.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Invalid image file or corrupted data.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.medianBlur(binary, 3)
        return denoised
    except Exception as e:
        logging.error(f"Error in preprocess_invoice: {e}")
        raise

def clean_text(text):
    """
    Cleans extracted text by removing unwanted characters and formatting numbers correctly.
    """
    text = text.replace(',', '')  # Remove commas from numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def extract_invoice_data(image_path):
    """
    Extracts structured data from invoice image using OCR.
    """
    try:
        img = preprocess_invoice(image_path)
        text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')

        lines = [clean_text(line) for line in text.split('\n') if line.strip()]
        
        logging.info("Extracted OCR Text:")
        for line in lines:
            logging.info(repr(line))

        extracted_data = []
        
        # Updated regex to capture invoice line items
        pattern = re.compile(r'([A-Za-z0-9\-\(\)]+)\s+\$?([\d,]+\.\d{2})\s+(\d+)\s+\$?([\d,]+\.\d{2})')

        for line in lines:
            match = pattern.search(line)
            if match:
                description = match.group(1).strip()
                rate = clean_text(match.group(2))
                qty = match.group(3).strip()
                line_total = clean_text(match.group(4))
                extracted_data.append([description, float(rate), int(qty), float(line_total)])

        if not extracted_data:
            raise ValueError("No structured data found. Please check the invoice format.")

        df = pd.DataFrame(extracted_data, columns=["Description", "Rate", "Qty", "Line Total"])
        return df
    except Exception as e:
        logging.error(f"Error in extract_invoice_data: {e}")
        return pd.DataFrame()

def process_invoice(file):
    """
    Processes the uploaded invoice file and extracts structured data.
    """
    try:
        temp_path = "temp_invoice.png"
        with open(temp_path, "wb") as f:
            f.write(file.read())

        if not file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            os.remove(temp_path)
            raise ValueError("Unsupported file format. Please upload PNG or JPG.")

        invoice_df = extract_invoice_data(temp_path)
        os.remove(temp_path)

        if invoice_df.empty:
            raise ValueError("No text extracted. Please check the image quality.")
        
        return invoice_df  # Return only the DataFrame
    except Exception as e:
        logging.error(f"Error processing invoice: {e}")
        raise ValueError(f"Error processing invoice: {e}")
