import streamlit as st
import pandas as pd
import plotly.express as px

# Import functions from backend modules
from semi_supervised.api_data import fetch_financial_data  # Semi-supervised model
from supervised.invoices import process_invoice        # Supervised model
from supervised.payslips import process_payslips          # Supervised model
from supervised.profit_loss import process_profit_loss    # Supervised model
from unsupervised.bart_classification import classify_data, categorize_data_with_kmeans # Unsupervised model

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

# Function to display visualizations
def display_visualizations(data, title):
    if isinstance(data, pd.DataFrame):
        data_df = data
    elif isinstance(data, dict):
        data_df = pd.DataFrame(list(data.items()), columns=["Field", "Value"])
    else:
        st.error("❌ Unable to generate visualizations: Unsupported data format.")
        return

    st.markdown(f"### {title} Visualizations")
    
    if "Description" in data_df.columns and "Amount" in data_df.columns:
        x_column, y_column = "Description", "Amount"
    elif "Field" in data_df.columns and "Value" in data_df.columns:
        x_column, y_column = "Field", "Value"
    else:
        st.error("❌ Unable to generate visualizations: Missing required columns.")
        return

    st.subheader("Bar Chart")
    fig_bar = px.bar(
        data_df, 
        x=x_column, 
        y=y_column, 
        text=y_column, 
        title=f"{title} Summary (Bar Chart)",
        color=x_column,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_bar.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.8)
    st.plotly_chart(fig_bar)

    st.subheader("Pie Chart")
    fig_pie = px.pie(
        data_df, 
        names=x_column, 
        values=y_column, 
        title=f"{title} Distribution (Pie Chart)",
        color_discrete_sequence=px.colors.sequential.Rainbow
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', pull=[0.1] * len(data_df))
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
                    st.success("✅ Invoice processed successfully!")
                    st.markdown("### 📄 Extracted Fields:")
                    st.write(invoice_data)
                    display_visualizations(invoice_data, "Invoice Data")
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
                st.markdown("### 📄 Classification Results:")
                st.write(classification_results)

                # Convert classification results to DataFrame for visualization
                classification_df = pd.DataFrame(classification_results)

                # Display visualizations
                display_visualizations(classification_df, "BART Classification")

                # Optional: Add K-Means clustering for further analysis
                if st.checkbox("Apply K-Means Clustering"):
                    n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
                    clustered_data = categorize_data_with_kmeans(classification_df, n_clusters=n_clusters)
                    st.success("✅ Clustering applied successfully!")
                    st.write(clustered_data)
                    display_visualizations(clustered_data, "Clustered Data")

            except Exception as e:
                st.error(f"❌ Error classifying data: {e}")
    
    elif category == "Semi-supervised":
        st.subheader("📊 Extract Financial Data from APIs")
        filing_url = st.text_input("Enter SEC Filing URL", "https://www.sec.gov/Archives/edgar/data/320193/000032019321000056/aapl-20210327.htm")
        if st.button("🔍 Fetch Financial Data"):
            try:
                data = fetch_financial_data(filing_url)
                st.success("✅ Financial data fetched successfully!")
                st.json(data)
                display_visualizations(data, "Financial Data")
            except Exception as e:
                st.error(f"❌ Error fetching financial data: {e}")

if __name__ == "__main__":
    main()
