import streamlit as st

# Predefined credentials
credentials = {
    'Ajay': 'starwars',
    'Guna': 'password',
    'Bhuvi': '876543210'
}

# Function to verify credentials
def verify_credentials(username, password):
    if username in credentials and credentials[username] == password:
        return True
    return False


# Streamlit login page
def login():
    st.title("Login Page")

    # Input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Login button
    if st.button("Login"):
        if verify_credentials(username, password):
            st.session_state.login_status = True
            st.session_state.username = username
            st.success(f"Welcome {username}!")
        else:
            st.error("Invalid username or password")
