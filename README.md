# Conversational Chatbot README

## Overview

This repository contains a conversational chatbot built using the StreamLit web framework and LangChain library. The chatbot is designed to interact with users through a web interface, leveraging various large language models (LLMs) for generating responses. It includes functionality for retrieving and re-ranking documents to provide more contextually accurate answers.

## Features

- **StreamLit Web API**: Provides an endpoint for chat interactions.
- **LangChain Integration**: Utilizes LangChain for chaining multiple models and processes.
- **Support for Multiple LLMs**: Easily switch between different LLMs including Falcon, LLaMA2, and GPT.
- **Document Retrieval Augmentation and Re-ranking**: Enhances response accuracy by fetching and ranking relevant documents.
- **Persistence and Configuration Options**: Customizable configurations for model loading and persistence.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ronair212/RAGChatBot.git
    cd RAGChatBot
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Environment Variables**:
    ```bash
    export COHERE_API_KEY="your-cohere-api-key"
    export OPENAI_API_KEY="your-openai-api-key"
    export TRANSFORMERS_CACHE="/path/to/models"
    export PYTORCH_TRANSFORMERS_CACHE="/path/to/models"
    export HUGGINGFACE_HUB_CACHE="/path/to/models"
    ```

### Configuration

The default configurations for the chatbot are stored in the `config.py` file. You can override these configurations using environment variables or command-line arguments.

### Running the Chatbot

1. **Run the Flask application**:
    ```bash
    python app.py
    ```

2. **Access the Chatbot**:
    

### Command-Line Arguments

You can customize the chatbot's behavior using command-line arguments:

```bash
python main.py --persist_directory /path/to/persist --load_in_8bit True --llm "tiiuae/falcon-7b" --rephrase True
```

## Supported LLMs

- `OpenAI GPT 3.5 Turbo`
- `tiiuae/falcon-7b`
- `tiiuae/falcon-40b`
- `tiiuae/falcon-7b-instruct`
- `tiiuae/falcon-40b-instruct`
- `upstage/Llama-2-70b-instruct`
- `togethercomputer/Llama-2-7B-32K-Instruct`

## Usage

### Chat Endpoint

- **Endpoint**: `/chat`
- **Method**: `POST`
- **Request Body**: JSON containing the user's message.
    ```json
    {
        "message": "Hello, how are you?"
    }
    ```

- **Response**: JSON containing the chatbot's response.
    ```json
    {
        "answer": "I'm good, thank you! How can I help you today?"
    }
    ```

### Example Usage

```python
import requests

response = requests.post('http://127.0.0.1:5000/chat', json={'message': 'Hello'})
print(response.json())
```

## Development

### Project Structure

- `app.py`: Main StreamLit application file.
- `chain.py`: Contains functions and classes related to LangChain initialization and operations.
- `config.py`: Configuration file for setting default parameters and options.
- `requirements.txt`: Python dependencies.
- `main.py`: Command-line interface for running the chatbot with custom configurations.

### Adding a New LLM

To add a new LLM:

1. **Define the Model in `config.py`**:
    ```python
    NEW_LLM_MODEL = "path/to/new-llm"
    ```

2. **Update `model_utils.py`** to include a function to initialize the new LLM.

3. **Modify `get_llm_pipeline` function** in `chain.py` to include a case for the new LLM.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
