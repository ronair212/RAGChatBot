import os
import streamlit as st
import chain
from config import (DEFAULT_CONFIG, LLM_FALCON_40B, LLM_FALCON_7B_INSTRUCT,
                    LLM_FALCON_40B_INSTRUCT, LLM_FALCON_7B, LLM_FALCON_40B,
                    LLM_LLAMA2_70B_INSTRUCT, LLM_LLAMA2_7B_32K_INSTRUCT, GPT)
from config import *

# Streamlit UI
st.title("Conversational Retrieval Chain")

# Input fields
persist_directory = st.text_input("Persist Directory", value=DEFAULT_CONFIG["persist_directory"])
quantization = st.selectbox("Quantization", options=["True", "False"], index=0 if DEFAULT_CONFIG["quantization"] else 1)
load_in_8bit = st.selectbox("Load in 8-bit", options=["True", "False"], index=0 if DEFAULT_CONFIG["load_in_8bit"] else 1)
llm = st.selectbox("LLM", options=[LLM_FALCON_7B_INSTRUCT, LLM_FALCON_40B_INSTRUCT, LLM_FALCON_7B, LLM_FALCON_40B, LLM_LLAMA2_70B_INSTRUCT, LLM_LLAMA2_7B_32K_INSTRUCT, LLM_FALCON_40B, GPT], index=0)
rephrase = st.selectbox("Rephrase", options=["True", "False"], index=0 if DEFAULT_CONFIG["rephrase"] else 1)

# Update DEFAULT_CONFIG based on user input
DEFAULT_CONFIG["persist_directory"] = persist_directory
DEFAULT_CONFIG["quantization"] = quantization
DEFAULT_CONFIG["load_in_8bit"] = load_in_8bit.lower() in ('yes', 'true', 't', 'y', '1')
DEFAULT_CONFIG["llm"] = llm
DEFAULT_CONFIG["rephrase"] = rephrase.lower() in ('yes', 'true', 't', 'y', '1')

# API Key input for GPT
if llm == "GPT":
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.error("OpenAI API Key is required for GPT.")
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize the conversational retrieval chain
if st.button("Initialize Chain"):
    retrieval_chain = chain.initialize_chain()
    st.session_state["retrieval_chain"] = retrieval_chain
    st.session_state["chat_history"] = [{"role": "assistant", "content": "Hey there, I am a chat assistant. What can I help you with today?"}]
    st.session_state["chat_active"] = True

# Chat interface
if "retrieval_chain" in st.session_state:
    st.subheader("Chat with Assistant")

    # Display chat history
    for message in st.session_state["chat_history"]:
        st.write(f"{message['role'].capitalize()}: {message['content']}")

    def send_message():
        user_input = st.session_state.user_input.strip()
        if user_input.lower() in ["exit", "quit", "bye", "thanks", "thnx", "thank you"]:
            st.session_state["chat_history"].append({"role": "assistant", "content": "Goodbye!"})
            st.session_state["chat_active"] = False
        elif user_input == "":
            st.session_state["chat_history"].append({"role": "assistant", "content": "What can I help you with?"})
        else:
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            assistant_response = st.session_state["retrieval_chain"]({"question": user_input})
            st.session_state["chat_history"].append({"role": "assistant", "content": assistant_response['answer']})
        st.session_state.user_input = ""  # Clear the text input
        st.experimental_rerun()  # Rerun to refresh the UI

    # Text input for user message
    if st.session_state.get("chat_active", True):
        st.text_input("User:", key="user_input", on_change=send_message)
        if st.button("Send"):
            send_message()
