import pytesseract
import cv2
import numpy as np  # Import numpy to resolve the error
import matplotlib.pyplot as plt
from io import BytesIO

def process_payslips(uploaded_payslip):
    try:
        # Convert the uploaded payslip file to an image
        img = cv2.imdecode(np.frombuffer(uploaded_payslip.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Extract text from the payslip
        text = pytesseract.image_to_string(img)
        
        # Generate a word frequency dictionary
        word_freq = {}
        for word in text.split():
            word = word.lower()  # Convert to lowercase for consistency
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort word frequencies in descending order
        sorted_word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])  # Top 10 words
        
        # Generate a bar plot for word frequencies
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(sorted_word_freq.keys(), sorted_word_freq.values(), color='skyblue')
        ax.set_title("Top  Frequencies in Payslip", fontsize=16)
        ax.set_xlabel("Fields", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Return the extracted text and the generated plot with the correct key
        return {'ExtractedText': text}, fig

    except Exception as e:
        # Handle errors gracefully
        return {'error': f"An error occurred: {str(e)}"}, None
