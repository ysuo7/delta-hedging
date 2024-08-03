import streamlit as st
from datetime import datetime
from delta_hedging_model import hedging
import pandas as pd

st.set_page_config(
    page_title="friend.tech Dashboard",
    page_icon="🥭",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; font-size: 100px;'>🥭🍍🍈🫕</h1>", unsafe_allow_html=True)

# Input for stock name
stock_name = st.text_input("Enter a stock name:")

# Input for strike price
strike_price = st.number_input("Enter the strike price:", value=100, step=1)

# Get current date
current_date = datetime.now()

# First Datetime Input: Start of the range
start_date = st.date_input(
    "Select the maturity start date:",
    value=current_date,
    min_value=pd.to_datetime('2023-01-01'),
    max_value=current_date,
    key="start_date"
)

# Second Datetime Input: End of the range
end_date = st.date_input(
    "Select the maturity end date:",
    value=current_date,
    min_value=start_date,
    key="end_date"
)

# Validate the input
if start_date > end_date:
    st.error("Start date cannot be later than end date.")
else:
    st.success("Datetime range is valid.")

# Display the selected input details in a single row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("**Selected stock name:**")
    st.write(stock_name)

with col2:
    st.write("**Selected strike price:**")
    st.write(strike_price)

with col3:
    st.write("**Selected start datetime:**")
    st.write(start_date)

with col4:
    st.write("**Selected end datetime:**")
    st.write(end_date)

# Button to trigger the hedging calculation
if st.button("Analyze"):
    # Call the hedging function when the button is clicked
    result = hedging(stock_name, start_date, end_date, strike_price)
    
    # Display the hedging result
    st.write("## Delta Hedging Result")
    st.write(result)
