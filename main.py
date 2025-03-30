# main.py

import streamlit as st
from login import login, verify_credentials
import demo3
import pandas as pd

# Initialize session state for login status
if 'login_status' not in st.session_state:
    st.session_state.login_status = False

# Page routing based on login status
if st.session_state.login_status:
    demo3.main_app()  # Directly show the forecasting app if logged in
else:
    login()  # Show login page if not logged in
