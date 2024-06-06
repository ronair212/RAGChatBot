
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from config import CACHE_DIR, LLM_FALCON_40B, LLM_LLAMA2_70B_INSTRUCT , LLM_LLAMA2_7B_32K_INSTRUCT
from config import *
from config import MODEL_UTILS_CONFIG
from transformers import BitsAndBytesConfig
from peft import PeftConfig , PeftModel

from transformers import BitsAndBytesConfig

def bit4quantization():

    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
    return quantization_config

def load_local_finetuned_falcon40binstruct_model():
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, #4bit quantizaition - load_in_4bit is used to load models in 4-bit quantization 
    bnb_4bit_use_double_quant=True, #nested quantization technique for even greater memory efficiency without sacrificing performance. This technique has proven beneficial, especially when fine-tuning large models
    bnb_4bit_quant_type="nf4", #quantization type used is 4 bit Normal Float Quantization- The NF4 data type is designed for weights initialized using a normal distribution
    bnb_4bit_compute_dtype=torch.bfloat16, #modify the data type used during computation. This can result in speed improvements. 
    )

    model_dir = "/work/LitArt/nair/chatbot_files/Falcon40b_RCDocs/Falcon40b-instruct_RCdataset-trained-model/"
    config = PeftConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir='/work/LitArt/nair/chatbot_files/models',
        )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model =  PeftModel.from_pretrained(model, model_dir)

    generation_config = model.generation_config
    generation_config.max_new_tokens = 200
    generation_config.temperature = 0.7
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    
    
    # Assuming the pipeline is the same for this model as others (adjust if not)
    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        config=generation_config,
        do_sample = True,
    )

    return hf_pipeline


    

def create_llama2_70b_instruct():
        config = MODEL_UTILS_CONFIG['create_llama2_70b_instruct']
        model_name = LLM_LLAMA2_70B_INSTRUCT
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/work/LitArt/nair/chatbot_files/models")
        tokenizer = AutoTokenizer.from_pretrained(model_name , cache_dir="/work/LitArt/nair/chatbot_files/models")
        hf_pipeline = pipeline(
                task="text-generation",
                model = model,
                do_sample=config['do_sample'], #Whether or not to use sampling ; use greedy decoding otherwise.
                tokenizer = tokenizer,
                max_new_tokens=config['max_new_tokens'], #The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
                model_kwargs=config['model_kwargs']
            )
        return hf_pipeline
    
    
    
    
def create_llama2_7b_32k_instruct():
        config = MODEL_UTILS_CONFIG['create_llama2_7b_32k_instruct']
        model_name = LLM_LLAMA2_7B_32K_INSTRUCT
        
        # Prepare common arguments for from_pretrained
        from_pretrained_args = {
            "pretrained_model_name_or_path": model_name,
            "cache_dir": "/work/LitArt/nair/chatbot_files/models",
        }
    
    
        # Add quantization_config if quantization is enabled
        if DEFAULT_CONFIG["quantization"]:
            from_pretrained_args["quantization_config"] = quantization_config
            from_pretrained_args["device_map"] = "auto"
        
        # Create the model using dynamic arguments
        model = AutoModelForCausalLM.from_pretrained(**from_pretrained_args)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name , cache_dir="/work/LitArt/nair/chatbot_files/models")
        hf_pipeline = pipeline(
                task="text-generation",
                model = model,
                do_sample=config['do_sample'], #Whether or not to use sampling ; use greedy decoding otherwise.
                tokenizer = tokenizer,
                max_new_tokens=config['max_new_tokens'], #The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
                model_kwargs=config['model_kwargs'],
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        return hf_pipeline

#create_llama2_7b_32k_instruct

def create_falcon_40b():
        quantization_config = bit4quantization()
        config = MODEL_UTILS_CONFIG['create_falcon_40b']
        model_name = LLM_FALCON_40B
        
        # Prepare common arguments for from_pretrained
        from_pretrained_args = {
            "pretrained_model_name_or_path": model_name,
            "cache_dir": "/work/LitArt/nair/chatbot_files/models",
        }
    
    
        # Add quantization_config if quantization is enabled
        if DEFAULT_CONFIG["quantization"]:
            from_pretrained_args["quantization_config"] = quantization_config
            from_pretrained_args["device_map"] = "auto"
        
        # Create the model using dynamic arguments
        model = AutoModelForCausalLM.from_pretrained(**from_pretrained_args)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name , cache_dir="/work/LitArt/nair/chatbot_files/models")
        hf_pipeline = pipeline(
                task="text-generation",
                model = model,
                do_sample=config['do_sample'], #Whether or not to use sampling ; use greedy decoding otherwise.
                tokenizer = tokenizer,
                max_new_tokens=config['max_new_tokens'], #The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
                model_kwargs=config['model_kwargs']
            )
        return hf_pipeline





def create_falcon_40b_instruct():
        quantization_config = bit4quantization()
        config = MODEL_UTILS_CONFIG['create_falcon_40b_instruct']
        model_name = LLM_FALCON_40B_INSTRUCT
        

        # Prepare common arguments for from_pretrained
        from_pretrained_args = {
            "pretrained_model_name_or_path": model_name,
            "cache_dir": "/work/LitArt/nair/chatbot_files/models",
        }
    
    
        # Add quantization_config if quantization is enabled
        if DEFAULT_CONFIG["quantization"]:
            from_pretrained_args["quantization_config"] = quantization_config
            from_pretrained_args["device_map"] = "auto"
        
        # Create the model using dynamic arguments
        model = AutoModelForCausalLM.from_pretrained(**from_pretrained_args)
        
        
        tokenizer = AutoTokenizer.from_pretrained(model_name , cache_dir="/work/LitArt/nair/chatbot_files/models")
        hf_pipeline = pipeline(
                task="text-generation",
                model = model,
                do_sample=config['do_sample'], #Whether or not to use sampling ; use greedy decoding otherwise.
                tokenizer = tokenizer,
                max_new_tokens=config['max_new_tokens'], #The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
                model_kwargs=config['model_kwargs'],
                #use_cache=True,#new
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                
            )
        return hf_pipeline
'''
pipeline = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=296,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)
'''



def create_falcon_7b():
        
        quantization_config = bit4quantization()
        



        config = MODEL_UTILS_CONFIG['create_falcon_7b']
        model_name = LLM_FALCON_7B
        
        # Prepare common arguments for from_pretrained
        from_pretrained_args = {
            "pretrained_model_name_or_path": model_name,
            "cache_dir": "/work/LitArt/nair/chatbot_files/models",
        }
    
    
        # Add quantization_config if quantization is enabled
        if DEFAULT_CONFIG["quantization"]:
            from_pretrained_args["quantization_config"] = quantization_config
            from_pretrained_args["device_map"] = "auto"
        
        # Create the model using dynamic arguments
        model = AutoModelForCausalLM.from_pretrained(**from_pretrained_args)
    
        tokenizer = AutoTokenizer.from_pretrained(model_name , cache_dir="/work/LitArt/nair/chatbot_files/models")
        hf_pipeline = pipeline(
                task="text-generation",
                model = model,
                do_sample=config['do_sample'], #Whether or not to use sampling ; use greedy decoding otherwise.
                tokenizer = tokenizer,
                max_new_tokens=config['max_new_tokens'], #The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
                model_kwargs=config['model_kwargs']
            )
        return hf_pipeline





def create_falcon_7b_instruct():
        quantization_config = bit4quantization()
        config = MODEL_UTILS_CONFIG['create_falcon_7b_instruct']
        model_name = LLM_FALCON_7B_INSTRUCT
        
        # Prepare common arguments for from_pretrained
        from_pretrained_args = {
            "pretrained_model_name_or_path": model_name,
            "cache_dir": "/work/LitArt/nair/chatbot_files/models",
        }
    
    
        # Add quantization_config if quantization is enabled
        if DEFAULT_CONFIG["quantization"]:
            from_pretrained_args["quantization_config"] = quantization_config
            from_pretrained_args["device_map"] = "auto"
        
        # Create the model using dynamic arguments
        model = AutoModelForCausalLM.from_pretrained(**from_pretrained_args)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name , cache_dir="/work/LitArt/nair/chatbot_files/models")
        hf_pipeline = pipeline(
                task="text-generation",
                model = model,
                do_sample=config['do_sample'], #Whether or not to use sampling ; use greedy decoding otherwise.
                tokenizer = tokenizer,
                max_new_tokens=config['max_new_tokens'], #The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
                model_kwargs=config['model_kwargs']
            )
        return hf_pipeline







def create_flan_t5_base(load_in_8bit=True):
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="/work/LitArt/nair/chatbot_files/models")
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
        
def gpt():
        from langchain.callbacks import StdOutCallbackHandler
        import os
        import getpass
        from langchain.chat_models import ChatOpenAI
        #os.environ["OPENAI_API_KEY"] = "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"
        context_callback = StdOutCallbackHandler()

        hf_pipeline = ChatOpenAI(temperature=0.5, verbose=True, openai_api_key = os.environ.get("OPENAI_API_KEY"), streaming=True, callbacks=[context_callback])
        
            
        
        return hf_pipeline


    

def load_huggingface_embeddings():
    """
    Loads the HuggingFaceEmbeddings.

    Returns:
        langchain.embeddings.HuggingFaceEmbeddings: The loaded embeddings.
    """
    return HuggingFaceEmbeddings()
