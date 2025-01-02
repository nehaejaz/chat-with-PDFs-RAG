import streamlit as st
from run_rag_pipeline import run_rag_pipeline
from streamlit_pdf_viewer import pdf_viewer
from create_db import generate_data_store


st.title("Chatbot")
st.write("This is a simple chatbot app. Ask me anything!")

# Initialize schema_name in session state
if 'schema_name' not in st.session_state:
    st.session_state['schema_name'] = ""

# Initialize uploaded_docs in session state
if 'uploaded_docs' not in st.session_state:
    st.session_state['uploaded_docs'] = None

# Placeholder to store conversation history
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Function to process uploaded documents
with st.sidebar:
    st.subheader("Your documents")
    docs = st.file_uploader("Upload your PDF here to start chatting", accept_multiple_files=True)
    if docs and docs != st.session_state['uploaded_docs']:
        st.session_state['uploaded_docs'] = docs  # Save uploaded documents in session state
        with st.spinner("Processing"):
            st.session_state['schema_name'] = generate_data_store(docs, "pdf")

# Function to get response from OpenAI
def get_chatbot_response(user_query):
    formatted_response = run_rag_pipeline(st.session_state['schema_name'], user_query)
    return formatted_response

chat_container = st.container()

# Display the conversation history
def display_chat_history():
    with chat_container:
        if st.session_state['conversation']:
            for msg in st.session_state['conversation']:
                if msg.startswith("You:"):
                    st.markdown(f"<div style='display: flex; justify-content: right; margin-top: 2%; margin-bottom:2%;'><div style='text-align: right; color:white; background-color: #007AFF; border-radius: 5px; padding: 1%; display: inline-block;'>{msg}</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align: left; color:white; background-color: #1f1b1b; border-radius: 5px; padding: 1%; display: inline-block; margin-top: 2%; margin-bottom:2%;'>{msg}</div>", unsafe_allow_html=True)

# Callback function to handle input and clear the input box
def handle_input():
    user_query = st.session_state.input_box  # Access the input value
    if user_query:
        # Add user query to conversation history
        st.session_state['conversation'].append(f"You: {user_query}")
        # Get chatbot response
        chatbot_response = get_chatbot_response(user_query)
        st.session_state['conversation'].append(f"{chatbot_response['text']}")
        # Clear the input box
        st.session_state.input_box = ""

# Get user input with callback
st.text_input(
    "Type your msg here:",
    key="input_box",
    on_change=handle_input,  # Trigger when the user presses Enter
)

# Display the conversation history
display_chat_history()
