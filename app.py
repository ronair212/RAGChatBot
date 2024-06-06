import os
import getpass
import streamlit as st
import chain
from config import (DEFAULT_CONFIG, LLM_FALCON_40B, LLM_FALCON_7B_INSTRUCT,
                    LLM_FALCON_40B_INSTRUCT, LLM_FALCON_7B, LLM_FALCON_40B,
                    LLM_LLAMA2_70B_INSTRUCT, LLM_LLAMA2_7B_32K_INSTRUCT, GPT)

def initialize_chain(llm):
    DEFAULT_CONFIG["llm"] = llm
    return chain.initialize_chain()

st.title("Conversational Retrieval Chain")

llm_options = [
    "LLM_FALCON_7B_INSTRUCT", 
    "LLM_FALCON_40B_INSTRUCT", 
    "LLM_FALCON_7B", 
    "LLM_FALCON_40B", 
    "LLM_LLAMA2_70B_INSTRUCT",
    "LLM_LLAMA2_7B_32K_INSTRUCT", 
    "GPT"
]

st.sidebar.title("Settings")
selected_llm = st.sidebar.selectbox("Select the LLM to use", llm_options)
rephrase = st.sidebar.checkbox("Rephrase question", value=DEFAULT_CONFIG["rephrase"])
load_in_8bit = st.sidebar.checkbox("Load in 8-bit mode", value=DEFAULT_CONFIG["load_in_8bit"])
persist_directory = st.sidebar.text_input("Persist Directory", DEFAULT_CONFIG["persist_directory"])

if selected_llm == "GPT":
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    os.environ["OPENAI_API_KEY"] = openai_api_key

DEFAULT_CONFIG["persist_directory"] = persist_directory
DEFAULT_CONFIG["load_in_8bit"] = load_in_8bit
DEFAULT_CONFIG["rephrase"] = rephrase

# Initialize the conversational retrieval chain
retrieval_chain = initialize_chain(selected_llm)

st.session_state.messages = st.session_state.get("messages", [])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    start_time = time.time()
    assistant_response = retrieval_chain({"question": prompt})
    end_time = time.time()

    response = assistant_response['answer']
    time_taken = end_time - start_time

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
    st.write(f"Time taken: {time_taken:.2f} seconds")
