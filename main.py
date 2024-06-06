

import argparse
import chain
import os
import getpass
import time

cohere_api_key = os.environ.get('COHERE_API_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')
from config import (DEFAULT_CONFIG, LLM_FALCON_40B, LLM_FALCON_7B_INSTRUCT,
                    LLM_FALCON_40B_INSTRUCT, LLM_FALCON_7B, LLM_FALCON_40B,
                    LLM_LLAMA2_70B_INSTRUCT, LLM_LLAMA2_7B_32K_INSTRUCT, GPT)

from config import *
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')

def get_args():
    parser = argparse.ArgumentParser(description="Initialize and run the conversational retrieval chain.")
    
    parser.add_argument("--persist_directory", type=str, default=DEFAULT_CONFIG["persist_directory"],
                        help="Directory for persisting data.")
    parser.add_argument("--quantization", type=str, default=DEFAULT_CONFIG["quantization"],
                        help="Quantization required True/False. Default False")
    parser.add_argument("--load_in_8bit", type=str2bool, default=DEFAULT_CONFIG["load_in_8bit"],
                        help="Whether to load in 8-bit mode. Accepts: yes, true, t, y, 1")
    parser.add_argument("--llm", type=str, default=DEFAULT_CONFIG["llm"],
                        choices=[LLM_FALCON_7B_INSTRUCT, LLM_FALCON_40B_INSTRUCT, LLM_FALCON_7B, LLM_FALCON_40B, LLM_LLAMA2_70B_INSTRUCT,LLM_LLAMA2_7B_32K_INSTRUCT, LOCAL_FALCON_40B_INSTRUCT_MODEL, GPT],
                        help="Which LLM to use. Options - LLM_FALCON_7B_INSTRUCT, LLM_FALCON_40B_INSTRUCT, LLM_FALCON_7B, LLM_FALCON_40B, LLM_LLAMA2_70B_INSTRUCT,LLM_LLAMA2_7B_32K_INSTRUCT, LOCAL_FALCON_40B_INSTRUCT_MODEL,GPT")
    parser.add_argument("--rephrase", type=str2bool, default=DEFAULT_CONFIG["rephrase"],
                        help="Whether to rephrase the current question from user. Accepts: yes, true, t, y, 1 , True , T ")

    return parser.parse_args()

def main():
    args = get_args()
    if args.llm == "GPT":
        openaikey = getpass.getpass("OPENAI API Key:")
        os.environ["OPENAI_API_KEY"] = openaikey
        OPENAI_API_KEY = openaikey

    # Overriding default config based on CLI arguments
    DEFAULT_CONFIG["persist_directory"] = args.persist_directory
    DEFAULT_CONFIG["load_in_8bit"] = args.load_in_8bit
    DEFAULT_CONFIG["quantization"] = args.quantization
    DEFAULT_CONFIG["llm"] = args.llm
    DEFAULT_CONFIG["rephrase"] = args.rephrase
    
    
    # Initialize the conversational retrieval chain
    retrieval_chain = chain.initialize_chain()
    
    print("Assistant: Hey there, I am a chat assistant. What can I help you with today?")
    chat_history = []

    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit", "bye", "thanks", "thnx", "thank you"]:
            print("Assistant: Goodbye!")
            break
            
            
        # Check for blank input
        if query.strip() == "":
            print("Assistant: What can I help you with?")
            continue
            
            
        start_time = time.time()  # Start time before processing the query
        assistant_response = retrieval_chain({"question": query})
        end_time = time.time()  # End time after processing the query

        time_taken = end_time - start_time  # Calculate the time taken

        print(f"Assistant: {assistant_response['answer']}")
        print(f"Time taken: {time_taken:.2f} seconds")  # Print the time taken
    
if __name__ == "__main__":
    main()


