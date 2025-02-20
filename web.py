import streamlit as st
import requests

# Page 1: Welcome page
def welcome_page():
    st.markdown(
        """
        <style>
            .container {
                position: relative;
                height: 25vh;
                display: flex;
                justify-content: center;
                align-items: center;
                text-align: center;
            }
            .stmarkdown>h1{
                position: absolute;
                bottom: 100px;
                left: 178px;
                font-size: 4em;
                color: red;
            }
            .stButton>button {
                background: #263540;
                color: #5aaeaa;
                padding: 10px 20px;
                border-radius: 10px;
                border: none;
                transition: background-color 0.3s ease;
                width: 100%;
                max-width: 400px;
                margin-top: 1px;
            }
            .stButton>button:hover {
                background-color: #FFFFFF;
                color: #57c8be;
            }
            .stButton {
                display: flex;
                justify-content: center;
            }
            /* Custom style to remove top margin and padding */
            .main {
                padding-top: 0;
                margin-top: -20px;
            }
            /* Hide the stDecoration element */
            [data-testid="stDecoration"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown('<h1>Welcome to Train Recommendation</h1>', unsafe_allow_html=True)
    if st.button("Let's Get Started", key="welcome_start_button"):
        st.session_state.page = 'main_page'
        st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Page 2: Main page with card layout
def main_page():
    st.set_page_config(page_title="Chatbot with Sidebar", layout="wide")

    # Custom CSS for the new theme with glass effect on hover
    st.markdown(
        """
        <style>
            :root {
                --primary-color: #000000;
                --secondary-color: #d36593;
                --background-gradient: linear-gradient(to bottom, #000000, #d36593);
                --text-color: #5aaeaa;
                --card-color: #57c9bb;
                --card-hover-color: #bff9f7;
                --card-text-color: #000000;
                --input-color: #263540;
                --button-color: #263540;
                --button-hover-color: #FFFFFF;
                --font: 'Arial', sans-serif;
            }
            body {
                background: var(--background-gradient);
                color: var(--text-color);
                font-family: var(--font);
            }
            .card-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: center;
                align-items: center;
            }
            .card {
                background: var(--card-color);
                border-radius: 10px;
                padding: 20px;
                margin: 10px;
                transition: transform 0.3s ease, background-color 0.3s ease;
                color: var(--card-text-color);
                width: 300px;
                height: 350px;  /* Fixed height for uniformity */
                box-sizing: border-box;
                text-align: center;
                border: 1px solid #d36593;
                position: relative;
                opacity: 0.9;
            }
            .card:hover {
                transform: scale(1.05);
                background-color: var(--card-hover-color);
            }
            .card h3 {
                color: var(--card-text-color);
                font-size: 1.5em;
                margin-bottom: 0.5em;
            }
            .card p, .card a {
                color: var(--card-text-color);
                text-decoration: none;
                margin: 0.5em 0;
            }
            .card-icon {
                font-size: 2em;
                margin-bottom: 0.5em;
            }
            .stTextInput>div>div>input {
                background: var(--input-color);
                border: none;
                border-radius: 10px;
                padding: 10px;
                color: var(--text-color);
                width: 100% !important;
            }
            .stButton>button {
                background: var(--button-color);
                color: var(--text-color);
                padding: 10px 20px;
                border-radius: 10px;
                border: none;
                transition: background-color 0.3s ease;
                width: 100%;
                max-width: 400px;
            }
            .stButton>button:active {
                background-color: var(--button-hover-color);
                color: var(--text-color);
            }
            .st-emotion-cache-1dp5vir {
                background-image: var(--background-gradient);
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Local Train Recommender")

    # Sidebar for user input
    user_prompt = st.text_input("Enter your location and destination and your preferences:", key="user_prompt")
    response_generated = st.button("Get Response", key="get_response_button")

    if response_generated:
        response = requests.post('http://127.0.0.1:5000/ask', json={'question': user_prompt})
        if response.status_code == 200:
            response_data = response.json()
            if response_data.get('status') == 'success':
                recommendations = response_data.get('recommendations', [])
                if recommendations:
                    cols = st.columns(len(recommendations))
                    for idx, rec in enumerate(recommendations):
                        with cols[idx]:
                            st.markdown(
                                f"<div class='card'>"
                                f"<h3>{rec['train_name']} ({rec['train_number']})</h3>"
                                f"<p>Departure: {rec['departure_time']}</p>"
                                f"<p>Arrival: {rec['arrival_time']}</p>"
                                f"<p>Distance: {rec['distance']} km</p>"
                                f"<p>Day Count: {rec['day_count']}</p>"
                                f"<p>Live Status: <a href='{rec['live_status_link']}' target='_blank'>Check here</a></p>"
                                f"<p>Map Status: <a href='{rec['map_status_link']}' target='_blank'>Check here</a></p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                else:
                    st.markdown(f"*Response:* No trains found.")
            else:
                st.markdown(f"*Response:* {response_data.get('message', 'No response from the server.')}")
        else:
            st.error("Error: Could not get response from the backend.")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# Navigation
def main():
    if st.session_state.page == 'welcome':
        welcome_page()
    elif st.session_state.page == 'main_page':
        main_page()

if __name__ == '__main__':
    main()
