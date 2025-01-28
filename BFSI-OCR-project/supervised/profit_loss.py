import pytesseract
import cv2
import numpy as np  # Import numpy
import matplotlib.pyplot as plt

def process_profit_loss(uploaded_profit_loss):
    """
    Processes the uploaded profit & loss statement to extract text
    and generate a bar chart showing word frequency.

    Args:
        uploaded_profit_loss: The uploaded file object (PDF, JPG, or PNG).

    Returns:
        A dictionary containing the extracted text and a matplotlib figure.
    """
    try:
        # Convert the uploaded file to an image using OpenCV
        img = cv2.imdecode(np.frombuffer(uploaded_profit_loss.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Extract text from the profit-loss statement using Tesseract OCR
        text = pytesseract.image_to_string(img)
        
        # Create a word frequency dictionary
        word_freq = {}
        for word in text.split():
            word = word.lower().strip(",.!?")  # Normalize words by converting to lowercase
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort the word frequencies and limit to top 10 for better visualization
        sorted_word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Generate a bar chart for the word frequency
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(sorted_word_freq.keys(), sorted_word_freq.values(), color='skyblue')
        ax.set_title(' Frequency in Profit & Loss Statement', fontsize=14)
        ax.set_xlabel('Field', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Return the extracted text and the generated plot with the correct key
        return {'ExtractedText': text}, fig
    
    except Exception as e:
        raise ValueError(f"Error processing the Profit & Loss statement: {e}")
