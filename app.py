import streamlit as st
from run_rag_pipeline import run_rag_pipeline

# Title of the app
st.title("Chatbot")
st.write("This is a simple chatbot app. Ask me anything!")

# Placeholder to store conversation history
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Function to get response from OpenAI
def get_chatbot_response(user_query):
    formatted_response = run_rag_pipeline(user_query)
    return formatted_response

chat_container = st.container()
    
# Display the conversation history
def display_chat_history():
    with chat_container:
        if st.session_state['conversation']:
            for msg in st.session_state['conversation']:
                if msg.startswith("You:"):
                    st.markdown(f"<div style='display: flex; justify-content: right; margin-top: 2%; margin-botton:2%;'><div style='text-align: right; color:white; background-color: #007AFF; border-radius: 5px; padding: 1%; display: inline-block;'>{msg}</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align: left; color:white; background-color: #1f1b1b; border-radius: 5px; padding: 1%; display: inline-block; margin-top: 2%; margin-botton:2%;'>{msg}</div>", unsafe_allow_html=True)


# Get user input
user_query = st.text_input("Type your msg here:", "", key="input_box")

# Generate chatbot response when user submits input
if user_query:
    chatbot_response = get_chatbot_response(user_query)
    # Add user input and chatbot response to conversation history
    st.session_state['conversation'].append(f"You: {user_query}")
    st.session_state['conversation'].append(f"{chatbot_response['text']}")
    st.input_box= ""
    display_chat_history()