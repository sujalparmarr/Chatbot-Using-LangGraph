import streamlit as st
from backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid
import os


def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    st.session_state['chat_counter'] += 1

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'chat_counter' not in st.session_state:
    st.session_state['chat_counter'] = len(st.session_state['chat_threads']) + 1

add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************

st.sidebar.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap" rel="stylesheet">

    <h1 style="
        font-family: 'Orbitron', sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: #39ff14;
        text-shadow: 2px 2px 5px #000,
                     4px 4px 10px #0ff;
        ">
        ChatBot Using LangGraph
    </h1>
    """,
    unsafe_allow_html=True
)

# **************************************** PDF Upload Section *********************************
st.sidebar.header('Upload PDF')

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.sidebar.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            from backend import process_pdf
            num_chunks = process_pdf("temp_uploaded.pdf")
            st.sidebar.success(f"PDF processed! Created {num_chunks} chunks.")
        
        # Clean up temp file
        os.remove("temp_uploaded.pdf")

st.sidebar.markdown("---")

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for idx, thread_id in enumerate(st.session_state['chat_threads'][::-1]):
    chat_number = len(st.session_state['chat_threads']) - idx
    if st.sidebar.button(f'Chat {chat_number}', key=thread_id):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages

# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    #CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }

    # first add the message to message_history
    with st.chat_message('assistant'):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config= CONFIG,
                stream_mode= 'messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    
    
    
    
    
    
    
