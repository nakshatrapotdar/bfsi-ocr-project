
import pytesseract
import cv2
import numpy as np
import re
import pandas as pd

def process_profit_loss(uploaded_profit_loss):
    """
    Processes the uploaded Profit & Loss statement to extract financial data.

    Args:
        uploaded_profit_loss: The uploaded file object (PDF, JPG, or PNG).

    Returns:
        A DataFrame containing extracted financial data.
    """
    try:
        # Read the uploaded image using OpenCV
        img = cv2.imdecode(np.frombuffer(uploaded_profit_loss.read(), np.uint8), cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply thresholding to improve OCR
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Extract text from image using Tesseract OCR
        text = pytesseract.image_to_string(thresh)

        # Print extracted text for debugging
        print("Extracted OCR Text:\n", text)

        # Define regex patterns to extract financial values
        patterns = {
            "Gross Profit": r"Gross profit\s+([\d,]+)",
            "Dividends Received": r"Dividends received\s+([\d,]+)",
            "Profit on Sale of Machine": r"Profit on sale of machine\s+([\d,]+)",
            "Depreciation": r"Depreciation\s+\(?([\d,]+)\)?",  # Handles negative values with or without parentheses
            "Interest Expense": r"Interest expense\s+\(?([\d,]+)\)?",
            "Distribution and Admin Expenses": r"Distribution, administration and other expenses\s+\(?([\d,]+)\)?",
            "Taxation": r"Taxation\s+\(?([\d,]+)\)?",
            "Net Profit": r"Profit for the year after taxation\s+([\d,]+)"
        }

        # Extract financial data
        profit_loss_data = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).replace(",", "")  # Remove commas from numbers
                # Handle negative values (if enclosed in parentheses)
                if "(" in match.group(0) and ")" in match.group(0):
                    value = f"-{value}"
                profit_loss_data[key] = int(value)
            else:
                profit_loss_data[key] = None  # If not found, mark as None

        # Convert the dictionary to a DataFrame for visualization
        profit_loss_df = pd.DataFrame(list(profit_loss_data.items()), columns=["Field", "Value"])

        return profit_loss_df

    except Exception as e:
        return {'error': f"Error processing the Profit & Loss statement: {e}"}
