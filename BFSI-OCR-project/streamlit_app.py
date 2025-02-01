'''

import streamlit as st
import pandas as pd
import plotly.express as px

# Import functions from backend modules
from semi_supervisedmodel.api_data import fetch_financial_data  # Semi-supervised model
from supervised_model.invoices import process_invoice        # Supervised model
from supervised_model.payslips import process_payslips          # Supervised model
from supervised_model.profit_loss import process_profit_loss    # Supervised model
from unsupervised_model.bart_classification import classify_data # Unsupervised model

# Function to set background color using CSS
def set_background():
    """
    Sets a gradient background for the Streamlit app.
    """
    bg_css = """
    <style>
    body, .stApp {
        background: linear-gradient(to right,rgb(234, 235, 194),rgb(149, 185, 247));
        color: black;
        font-family: Arial, sans-serif;
    }
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
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# ✅ FIXED: Improved visualization handling
def display_visualizations(data, title):
    """
    Displays bar chart and pie chart visualizations for extracted data.
    """
    if isinstance(data, pd.DataFrame):  # ✅ Handle DataFrame directly
        data_df = data  # Already a DataFrame
    elif isinstance(data, dict):  # ✅ Convert dictionary to DataFrame
        data_df = pd.DataFrame(list(data.items()), columns=["Field", "Value"])
    else:
        st.error("❌ Unable to generate visualizations: Unsupported data format.")
        return

    st.markdown(f"### {title} Visualizations")
    
    # ✅ FIXED: Ensure proper column names for visualizations
    if "Description" in data_df.columns and "Line Total" in data_df.columns:
        x_column, y_column = "Description", "Line Total"
    elif "Field" in data_df.columns and "Value" in data_df.columns:
        x_column, y_column = "Field", "Value"
    else:
        st.error("❌ Unable to generate visualizations: Missing required columns.")
        return

    # Bar Chart
    fig_bar = px.bar(data_df, x=x_column, y=y_column, text=y_column, title=f"{title} Summary (Bar Chart)")
    st.plotly_chart(fig_bar)

    # Line Chart
    fig_line = px.line(data_df, x=x_column, y=y_column, markers=True, title=f"{title} Trend (Line Chart)")
    st.plotly_chart(fig_line)

    # Pie Chart
    fig_pie = px.pie(data_df, names=x_column, values=y_column, title=f"{title} Distribution (Pie Chart)")
    st.plotly_chart(fig_pie)

def main():
    set_background()
    st.title('📊 BFSI - OCR of Bank Statements')
    
    st.sidebar.title("🔍 Model Selection")
    category = st.sidebar.selectbox("Choose a category", ["Supervised", "Unsupervised", "Semi-supervised"])
    
    if category == "Supervised":
        sub_option = st.sidebar.radio("Choose a supervised model", ["Invoice Processing", "Payslip Processing", "Profit & Loss Processing"])
        
        if sub_option == "Invoice Processing":
            st.subheader("🧾 Process Invoices")
            uploaded_invoice = st.file_uploader("Upload Invoice", type=['pdf', 'jpg', 'png'])
            if uploaded_invoice is not None:
                try:
                    invoice_data = process_invoice(uploaded_invoice)

                    # ✅ FIXED: Ensure invoice data is a DataFrame
                    if isinstance(invoice_data, pd.DataFrame):
                        invoice_df = invoice_data
                    else:
                        raise ValueError("Unexpected data format returned by process_invoice.")

                    st.success("✅ Invoice processed successfully!")
                    st.markdown("### 📄 Extracted Fields:")
                    st.write(invoice_df)

                    display_visualizations(invoice_df, "Invoice Data")
                except Exception as e:
                    st.error(f"❌ Error processing invoice: {e}")
        
        elif sub_option == "Payslip Processing":
            st.subheader("💰 Process Payslips")
            uploaded_payslip = st.file_uploader("Upload Payslip", type=['pdf', 'jpg', 'png'])
            if uploaded_payslip is not None:
                try:
                    payslip_data = process_payslips(uploaded_payslip)
                    st.success("✅ Payslip processed successfully!")
                    st.write(payslip_data)
                    display_visualizations(payslip_data, "Payslip Data")
                except Exception as e:
                    st.error(f"❌ Error processing payslip: {e}")
        
        elif sub_option == "Profit & Loss Processing":
            st.subheader("📉 Process Profit & Loss Statements")
            uploaded_profit_loss = st.file_uploader("Upload Profit & Loss Statement", type=['pdf', 'jpg', 'png'])
            if uploaded_profit_loss is not None:
                try:
                    profit_loss_data = process_profit_loss(uploaded_profit_loss)
                    st.success("✅ Profit & Loss statement processed successfully!")
                    st.write(profit_loss_data)
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
                st.json(classification_results)
            except Exception as e:
                st.error(f"❌ Error classifying data: {e}")
    
    elif category == "Semi-supervised":
        st.subheader("📊 Extract Financial Data from APIs")
        filing_url = st.text_input("Enter SEC Filing URL", "https://www.sec.gov/Archives/edgar/data/320193/000032019321000056/aapl-20210327.htm")
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
    main()'''

import streamlit as st
import pandas as pd
import plotly.express as px

# Import functions from backend modules
from semi_supervised.api_data import fetch_financial_data  # Semi-supervised model
from supervised.invoices import process_invoice        # Supervised model
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
        background: linear-gradient(to right,rgb(234, 235, 194),rgb(149, 185, 247));
        color: black;
        font-family: Arial, sans-serif;
    }
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
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# ✅ FIXED: Improved visualization handling
def display_visualizations(data, title):
    """
    Displays bar chart, line chart, and pie chart visualizations for extracted data.
    """
    if isinstance(data, pd.DataFrame):  # ✅ Handle DataFrame directly
        data_df = data  # Already a DataFrame
    elif isinstance(data, dict):  # ✅ Convert dictionary to DataFrame
        data_df = pd.DataFrame(list(data.items()), columns=["Field", "Value"])
    else:
        st.error("❌ Unable to generate visualizations: Unsupported data format.")
        return

    st.markdown(f"### {title} Visualizations")
    
    # ✅ FIXED: Ensure proper column names for visualizations
    if "Description" in data_df.columns and "Line Total" in data_df.columns:
        x_column, y_column = "Description", "Line Total"
    elif "Field" in data_df.columns and "Value" in data_df.columns:
        x_column, y_column = "Field", "Value"
    else:
        st.error("❌ Unable to generate visualizations: Missing required columns.")
        return

    # Bar Chart with custom colors
    st.subheader("Bar Chart")
    fig_bar = px.bar(
        data_df, 
        x=x_column, 
        y=y_column, 
        text=y_column, 
        title=f"{title} Summary (Bar Chart)",
        color=x_column,  # Color by category
        color_discrete_sequence=px.colors.qualitative.Pastel  # Use a colorful palette
    )
    fig_bar.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.8)
    st.plotly_chart(fig_bar)

    # Line Chart with markers and custom colors
    st.subheader("Line Chart")
    fig_line = px.line(
        data_df, 
        x=x_column, 
        y=y_column, 
        markers=True, 
        title=f"{title} Trend (Line Chart)",
        color_discrete_sequence=px.colors.qualitative.Vivid  # Use a vibrant color palette
    )
    fig_line.update_traces(line=dict(width=2.5), marker=dict(size=10))
    st.plotly_chart(fig_line)

    # Pie Chart with custom colors
    st.subheader("Pie Chart")
    fig_pie = px.pie(
        data_df, 
        names=x_column, 
        values=y_column, 
        title=f"{title} Distribution (Pie Chart)",
        color_discrete_sequence=px.colors.sequential.Rainbow  # Use a rainbow color palette
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', pull=[0.1] * len(data_df))  # Add pull effect
    st.plotly_chart(fig_pie)

def main():
    set_background()
    st.title('📊 BFSI - OCR of Bank Statements')
    
    st.sidebar.title("🔍 Model Selection")
    category = st.sidebar.selectbox("Choose a category", ["Supervised", "Unsupervised", "Semi-supervised"])
    
    if category == "Supervised":
        sub_option = st.sidebar.radio("Choose a supervised model", ["Invoice Processing", "Payslip Processing", "Profit & Loss Processing"])
        
        if sub_option == "Invoice Processing":
            st.subheader("🧾 Process Invoices")
            uploaded_invoice = st.file_uploader("Upload Invoice", type=['pdf', 'jpg', 'png'])
            if uploaded_invoice is not None:
                try:
                    invoice_data = process_invoice(uploaded_invoice)

                    # ✅ FIXED: Ensure invoice data is a DataFrame
                    if isinstance(invoice_data, pd.DataFrame):
                        invoice_df = invoice_data
                    else:
                        raise ValueError("Unexpected data format returned by process_invoice.")

                    st.success("✅ Invoice processed successfully!")
                    st.markdown("### 📄 Extracted Fields:")
                    st.write(invoice_df)

                    display_visualizations(invoice_df, "Invoice Data")
                except Exception as e:
                    st.error(f"❌ Error processing invoice: {e}")
        
        elif sub_option == "Payslip Processing":
            st.subheader("💰 Process Payslips")
            uploaded_payslip = st.file_uploader("Upload Payslip", type=['pdf', 'jpg', 'png'])
            if uploaded_payslip is not None:
                try:
                    payslip_data = process_payslips(uploaded_payslip)
                    st.success("✅ Payslip processed successfully!")
                    st.write(payslip_data)
                    display_visualizations(payslip_data, "Payslip Data")
                except Exception as e:
                    st.error(f"❌ Error processing payslip: {e}")
        
        elif sub_option == "Profit & Loss Processing":
            st.subheader("📉 Process Profit & Loss Statements")
            uploaded_profit_loss = st.file_uploader("Upload Profit & Loss Statement", type=['pdf', 'jpg', 'png'])
            if uploaded_profit_loss is not None:
                try:
                    profit_loss_data = process_profit_loss(uploaded_profit_loss)
                    st.success("✅ Profit & Loss statement processed successfully!")
                    st.write(profit_loss_data)
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
                st.json(classification_results)
            except Exception as e:
                st.error(f"❌ Error classifying data: {e}")
    
    elif category == "Semi-supervised":
        st.subheader("📊 Extract Financial Data from APIs")
        filing_url = st.text_input("Enter SEC Filing URL", "https://www.sec.gov/Archives/edgar/data/320193/000032019321000056/aapl-20210327.htm")
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
