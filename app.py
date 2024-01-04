import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load('bankruptcy_classification_model.pkl')


# Function to classify Bankrupt or non-Bankrupt
def bankruptcy_classification(features):
    input_data = pd.DataFrame(features, index=[0])
    outputClass = model.predict(input_data)[0]
    return outputClass

# Loading the dataset
df = pd.read_csv('bankruptcy-prevention.csv',sep=';')
df.columns = df.columns.str.strip()

# Streamlit UI
st.markdown(
    """
    <div style="font-family: Times New Roman; font-size:50px;">
        Bankruptcy Prevention
    </div>
    """,
    unsafe_allow_html=True
)

# User input form
st.sidebar.header('Select Input Features: ')
industrial_risk = st.sidebar.slider('Industrial Risk', min_value=float(df['industrial_risk'].min()), max_value=float(df['industrial_risk'].max()), step=0.5)
management_risk = st.sidebar.slider('Management Risk', min_value=float(df['management_risk'].min()), max_value=float(df['management_risk'].max()), step=0.5)
financial_flexibility = st.sidebar.slider('Financial Flexibility', min_value=float(df['financial_flexibility'].min()), max_value=float(df['financial_flexibility'].max()), step=0.5)
credibility = st.sidebar.slider('Credibility', min_value=float(df['credibility'].min()), max_value=float(df['credibility'].max()), step=0.5)
competitiveness = st.sidebar.slider('Competitiveness', min_value=float(df['competitiveness'].min()), max_value=float(df['competitiveness'].max()), step=0.5)
operating_risk = st.sidebar.slider('Operating Risk', min_value=float(df['operating_risk'].min()), max_value=float(df['operating_risk'].max()), step=0.5)

# User input features
user_input = {
    'industrial_risk': industrial_risk,
    'management_risk': management_risk,
    'financial_flexibility': financial_flexibility,
    'credibility': credibility,
    'competitiveness': competitiveness,
    'operating_risk': operating_risk,
}

# Classification
if st.sidebar.button('Bankruptcy Prevention'):
    outputClass = bankruptcy_classification(user_input)
    if outputClass == 0:
        outputClass="Chances of Bankruptcy"
    else:
        outputClass = 'No chances of Bankruptcy'
    st.success(f'Classified as:  { outputClass }')

# Adding "Developed by Group - 1 (P318)" at the bottom right
st.markdown(
    """
    <div style="position: fixed; bottom: 15px; right: 20px; text-align: left; font-size:15px; font-family: Courier new">
        Developed by <span style="font-family: Times New Roman; font-size:25px; color:#ff4b4b">Group-2 (P318)</span>
    </div>
    """, 
    unsafe_allow_html=True
)