import pytesseract
import cv2
import numpy as np
import pandas as pd
import re

def extract_payslip_data(image):
    """
    Extracts structured data from a payslip image using OCR.
    """
    try:
        # Convert the uploaded payslip file to an image
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Convert to grayscale and apply threshold for better OCR accuracy
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Extract text from the payslip
        text = pytesseract.image_to_string(thresh)

        # Clean and structure the extracted text
        structured_data = {}

        # Extract specific fields using regex
        structured_data["Employee Name"] = re.search(r'EMPLOYEE NAME\s+([A-Za-z\s.]+)', text)
        structured_data["Employee ID"] = re.search(r'EMPLOYEE ID\s+(\d+)', text)
        structured_data["Check No"] = re.search(r'CHECK NO\.\s+(\d+)', text)
        structured_data["Pay Period"] = re.search(r'PAY PERIOD\s+([\d/-]+)', text)
        structured_data["Pay Date"] = re.search(r'PAY DATE\s+([\d/-]+)', text)

        structured_data["Gross Wages"] = re.search(r'GROSS WAGES\s+([\d,.]+)', text)
        structured_data["Net Pay"] = re.search(r'NET PAY\s+([\d,.]+)', text)

        structured_data["FICA MED Tax"] = re.search(r'FICA MED TAX\s+([\d,.]+)', text)
        structured_data["FICA SS Tax"] = re.search(r'FICA SS TAX\s+([\d,.]+)', text)
        structured_data["Federal Tax"] = re.search(r'FED TAX\s+([\d,.]+)', text)

        structured_data["YTD Gross"] = re.search(r'YTD GROSS\s+([\d,.]+)', text)
        structured_data["YTD Deductions"] = re.search(r'YTD DEDUCTIONS\s+([\d,.]+)', text)
        structured_data["YTD Net Pay"] = re.search(r'YTD NET PAY\s+([\d,.]+)', text)

        # Convert extracted values
        for key, match in structured_data.items():
            structured_data[key] = match.group(1).strip() if match else None
        
        # Convert numerical fields to float
        numeric_fields = ["Gross Wages", "Net Pay", "FICA MED Tax", "FICA SS Tax", "Federal Tax", "YTD Gross", "YTD Deductions", "YTD Net Pay"]
        for field in numeric_fields:
            if structured_data[field]:
                structured_data[field] = float(structured_data[field].replace(',', ''))

        return structured_data

    except Exception as e:
        return {'error': f"An error occurred while extracting data: {str(e)}"}

def process_payslips(uploaded_payslip):
    """
    Processes the uploaded payslip file and extracts structured data.
    """
    try:
        # Extract structured data from the payslip
        payslip_data = extract_payslip_data(uploaded_payslip)

        if 'error' in payslip_data:
            return payslip_data

        # Convert the structured data to a DataFrame for visualization
        payslip_df = pd.DataFrame(list(payslip_data.items()), columns=["Field", "Value"])

        return payslip_df
    except Exception as e:
        return {'error': f"An error occurred while processing payslip: {str(e)}"}
