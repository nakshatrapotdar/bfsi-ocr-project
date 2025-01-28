import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from sklearn.cluster import KMeans
import json

def classify_data(uploaded_data, transaction_id=None):
    try:
        # Read the data based on the file format
        if uploaded_data.name.endswith('.csv'):
            data = pd.read_csv(uploaded_data)
        elif uploaded_data.name.endswith('.json'):
            data = pd.read_json(uploaded_data)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or JSON file.")

        # Ensure essential columns exist
        required_columns = ["Transaction_ID", "Description", "Amount", "Date"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Fill missing values with placeholders
        data.fillna("Unknown", inplace=True)

        # Combine Description and Amount into a single text string for BART processing
        data["Text_Input"] = data["Description"] + " - " + data["Amount"].astype(str)

        # Initialize the BART model and tokenizer
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

        # Tokenize and generate summaries/classifications for each transaction
        inputs = tokenizer(list(data["Text_Input"]), return_tensors="pt", padding=True, truncation=True, max_length=512)
        summary = model.generate(inputs['input_ids'], max_length=150)

        # Decode the output summaries and prepare the classification result
        decoded_summary = [tokenizer.decode(s, skip_special_tokens=True) for s in summary]

        # Add the summary back to the data
        data["Classification_Result"] = decoded_summary

        # Query a specific Transaction ID if provided
        if transaction_id is not None:
            transaction_result = data[data["Transaction_ID"] == transaction_id]
            if not transaction_result.empty:
                return transaction_result[["Transaction_ID", "Classification_Result"]].to_dict(orient='records')
            else:
                return {"Error": f"No data found for Transaction ID: {transaction_id}"}

        # Return the entire classification result
        return data[["Transaction_ID", "Description", "Amount", "Classification_Result"]].to_dict(orient='records')

    except Exception as e:
        raise ValueError(f"Error processing file: {e}")


def categorize_data_with_kmeans(data, n_clusters=3):
    """
    Perform K-Means clustering to categorize transactions based on amount.

    Args:
        data (pd.DataFrame): Input transaction data.
        n_clusters (int): Number of clusters for K-Means.

    Returns:
        pd.DataFrame: Dataframe with an additional "Cluster" column.
    """
    try:
        # Ensure Amount column is numeric
        data["Amount"] = pd.to_numeric(data["Amount"], errors="coerce").fillna(0)

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data["Cluster"] = kmeans.fit_predict(data[["Amount"]])

        return data
    except Exception as e:
        raise ValueError(f"Error performing clustering: {e}")
