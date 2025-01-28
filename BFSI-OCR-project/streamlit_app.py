import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Import functions from backend modules
from semi_supervised.api_data import fetch_financial_data  # Semi-supervised model
from supervised.invoices import process_invoices          # Supervised model
from supervised.payslips import process_payslips          # Supervised model
from supervised.profit_loss import process_profit_loss    # Supervised model
from unsupervised.bart_classification import classify_data # Unsupervised model

# Function to set background color using CSS
def set_background():
    """
    Sets a gradient background for the Streamlit app.
    """
    bg_css = """
    <style>
    body, .stApp {
        background: linear-gradient(to right,rgb(234, 235, 194),rgb(149, 185, 247)); /* Blue gradient */
        color: white; /* Text color */
        font-family: Arial, sans-serif;
    }
    
    /* Style sidebar */
    .css-1lcbmhc, .css-18e3th9 { 
        background: rgba(0, 0, 0, 0.7); /* Dark sidebar */
        color: white;
    }

    /* Style buttons */
    .stButton>button {
        background-color: #ff9800;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }

    .stButton>button:hover {
        background-color: #ff5722;
    }

    /* Centered Title */
    .css-1aumxhk {
        text-align: center;
    }
    
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

def display_visualizations(data, title):
    """
    Generates bar and pie charts for the given data.

    Args:
        data: Data dictionary or text to visualize.
        title: Title for the visualizations.
    """
    if isinstance(data, dict) and 'ExtractedText' in data:
        extracted_text = data['ExtractedText']
        word_freq = {}
        for word in extracted_text.split():
            word = word.lower().strip(",.!?")  # Normalize words
            word_freq[word] = word_freq.get(word, 0) + 1
        sorted_word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])  # Top 10 words
        data_df = pd.DataFrame(list(sorted_word_freq.items()), columns=["Field", "Value"])
    elif isinstance(data, dict):
        data_df = pd.DataFrame(list(data.items()), columns=["Field", "Value"])
    else:
        st.error("Unable to generate visualizations: Unsupported data format.")
        return

    st.markdown(f"### {title} Visualizations")

    # Bar chart
    st.subheader("Bar Chart")
    fig_bar = px.bar(data_df, x="Field", y="Value", text="Value", title=f"Bar Chart of {title}")
    st.plotly_chart(fig_bar)

    # Pie chart
    st.subheader("Pie Chart")
    fig_pie = px.pie(data_df, names="Field", values="Value", title=f"Pie Chart of {title}")
    st.plotly_chart(fig_pie)

def main():
    # Apply background styling
    set_background()

    st.title('📊 BFSI - OCR of Bank Statements')
    st.markdown(
        """
        <style>
            .black-text {
                color: black;
            }
        </style>
        <div class="black-text">
            This application processes <b>bank statements, invoices, payslips,</b> and <b>profit & loss statements</b>.
            It also supports <b>classification of data using the BART model</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for navigation
    st.sidebar.title("🔍 Model Selection")
    category = st.sidebar.selectbox(
        "Choose a category", 
        ["Supervised", "Unsupervised", "Semi-supervised"]
    )

    if category == "Supervised":
        sub_option = st.sidebar.radio(
            "Choose a supervised model",
            ["Invoice Processing", "Payslip Processing", "Profit & Loss Processing"]
        )
        if sub_option == "Invoice Processing":
            st.subheader("🧾 Process Invoices")
            uploaded_invoice = st.file_uploader("Upload Invoice", type=['pdf', 'jpg', 'png'])
            if uploaded_invoice is not None:
                try:
                    invoice_data, fig = process_invoices(uploaded_invoice)
                    st.success("✅ Invoice processed successfully!")
                    st.markdown("### 📄 Extracted Fields:")
                    st.text_area("Extracted Text", invoice_data.get('ExtractedText', ''), height=300)
                    st.pyplot(fig)
                    display_visualizations(invoice_data, "Invoice Data")
                except Exception as e:
                    st.error(f"❌ Error processing invoice: {e}")

        elif sub_option == "Payslip Processing":
            st.subheader("💰 Process Payslips")
            uploaded_payslip = st.file_uploader("Upload Payslip", type=['pdf', 'jpg', 'png'])
            if uploaded_payslip is not None:
                try:
                    payslip_data, fig = process_payslips(uploaded_payslip)
                    st.success("✅ Payslip processed successfully!")
                    st.markdown("### 📄 Extracted Fields:")
                    st.text_area("Extracted Text", payslip_data.get('ExtractedText', ''), height=300)
                    st.pyplot(fig)
                    display_visualizations(payslip_data, "Payslip Data")
                except Exception as e:
                    st.error(f"❌ Error processing payslip: {e}")

        elif sub_option == "Profit & Loss Processing":
            st.subheader("📉 Process Profit & Loss Statements")
            uploaded_profit_loss = st.file_uploader("Upload Profit & Loss Statement", type=['pdf', 'jpg', 'png'])
            if uploaded_profit_loss is not None:
                try:
                    profit_loss_data, fig = process_profit_loss(uploaded_profit_loss)
                    st.success("✅ Profit & Loss statement processed successfully!")
                    st.markdown("### 📄 Extracted Fields:")
                    st.text_area("Extracted Text", profit_loss_data.get('ExtractedText', ''), height=300)
                    st.pyplot(fig)
                    display_visualizations(profit_loss_data, "Profit & Loss Data")
                except Exception as e:
                    st.error(f"❌ Error processing Profit & Loss statement: {e}")

    elif category == "Unsupervised":
        st.subheader("🤖 Classify Data Using BART Model")
        uploaded_data = st.file_uploader("Upload Data File", type=['csv', 'json'])
        if uploaded_data is not None:
            try:
                classification_results = classify_data(uploaded_data)
                st.success("✅ Data classified successfully!")
                st.markdown("### 📄 Classification Results:")
                st.json(classification_results)
            except Exception as e:
                st.error(f"❌ Error classifying data: {e}")

    elif category == "Semi-supervised":
        st.subheader("📊 Extract Financial Data from APIs")
        filing_url = st.text_input(
            "Enter SEC Filing URL", 
            "https://www.sec.gov/Archives/edgar/data/320193/000032019321000056/aapl-20210327.htm"
        )
        if st.button("🔍 Fetch Financial Data"):
            try:
                data = fetch_financial_data(filing_url)
                if 'error' in data:
                    st.error(data['error'])
                else:
                    st.success("✅ Financial data fetched successfully!")
                    st.json(data)
                    display_visualizations(data, "Financial Data")
            except Exception as e:
                st.error(f"❌ Error fetching financial data: {e}")

if __name__ == "__main__":
    main()
