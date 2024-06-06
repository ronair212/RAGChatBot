from flask import Flask, request, jsonify
import os
import chain
from config import DEFAULT_CONFIG, LLM_FALCON_40B_INSTRUCT, LLM_FALCON_7B, LLM_FALCON_40B, LLM_FALCON_7B_INSTRUCT, LLM_LLAMA2_70B_INSTRUCT, LLM_LLAMA2_7B_32K_INSTRUCT, GPT
import threading

app = Flask(__name__)

# Function to initialize the chatbot
def initialize_chatbot():
    #DEFAULT_CONFIG["persist_directory"] = persist_directory
    #DEFAULT_CONFIG["load_in_8bit"] = False
    DEFAULT_CONFIG["llm"] = "GPT"

    return chain.initialize_chain()

# Initialize the chatbot with default configurations
retrieval_chain = initialize_chatbot()

# Route for chatting
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')

    if user_message.lower() in ["exit", "quit", "bye", "thanks", "thnx"]:
        return jsonify({'answer': "Goodbye!"})

    assistant_response = retrieval_chain({"question": user_message})
    return jsonify({'answer': assistant_response['answer']})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
