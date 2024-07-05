import streamlit as st
import joblib
import pandas as pd

# Load the trained XGBoost model
model = joblib.load('xgboost_model.pkl')

# Define the prediction function
def predict_fraud(input_data):
    prediction = model.predict_proba(input_data)[:, 1]
    return prediction

# Streamlit app
st.title('Fraud Detection App')

st.header('Enter transaction details')

# Explanation for each input
st.markdown('**Step**: Time step of the transaction')
st.markdown('**Amount**: Transaction amount')
st.markdown('**Old Balance Orig**: Sender\'s account balance before transaction')
st.markdown('**New Balance Orig**: Sender\'s account balance after transaction')
st.markdown('**Old Balance Dest**: Receiver\'s account balance before transaction')
st.markdown('**New Balance Dest**: Receiver\'s account balance after transaction')
st.markdown('**Type CASH_OUT**: Check if transaction involves cash out (tick if yes)')
st.markdown('**Type TRANSFER**: Check if transaction involves transfer (tick if yes)')

# Collect user input
step = st.number_input('Step', min_value=0, value=0)
amount = st.number_input('Amount', min_value=0.0, value=0.0)
oldbalanceOrg = st.number_input('Old Balance Orig', min_value=0.0, value=0.0)
newbalanceOrig = st.number_input('New Balance Orig', min_value=0.0, value=0.0)
oldbalanceDest = st.number_input('Old Balance Dest', min_value=0.0, value=0.0)
newbalanceDest = st.number_input('New Balance Dest', min_value=0.0, value=0.0)
type_CASH_OUT = st.checkbox('Type: CASH_OUT')
type_TRANSFER = st.checkbox('Type: TRANSFER')

# Prepare the input for prediction
input_data = pd.DataFrame({
    'step': [step],
    'amount': [amount],
    'oldbalanceOrg': [oldbalanceOrg],
    'newbalanceOrig': [newbalanceOrig],
    'oldbalanceDest': [oldbalanceDest],
    'newbalanceDest': [newbalanceDest],
    'type_CASH_OUT': [int(type_CASH_OUT)],
    'type_TRANSFER': [int(type_TRANSFER)]
})

# Make prediction
if st.button('Predict'):
    prediction = predict_fraud(input_data)
    st.write(f'Fraud Probability: {prediction[0]:.2f}')
