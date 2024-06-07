# Constants related to environment variables

OPENAI_API_KEY = "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"
COHERE_API_KEY = "mVc5EhnC3eDPNFwV3d7ai8IudTmsfijju7h7oxQP"
TRANSFORMERS_CACHE = '/work/LitArt/nair/chatbot_files/models'
PYTORCH_TRANSFORMERS_CACHE = '/work/LitArt/nair/chatbot_files/models'
HUGGINGFACE_HUB_CACHE = '/work/LitArt/nair/chatbot_files/models'



# Hardcoded URLs for data loading
DATA_URLS = ["https://rc-docs.northeastern.edu/en/latest/welcome/index.html",
"https://rc-docs.northeastern.edu/en/latest/welcome/welcome.html",
"https://rc-docs.northeastern.edu/en/latest/welcome/services.html",
"https://rc-docs.northeastern.edu/en/latest/welcome/gettinghelp.html",
"https://rc-docs.northeastern.edu/en/latest/welcome/introtocluster.html",
"https://rc-docs.northeastern.edu/en/latest/welcome/casestudiesandtestimonials.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/index.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/get_access.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/accountmanager.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/connectingtocluster/index.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/connectingtocluster/mac.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/connectingtocluster/windows.html",
"https://rc-docs.northeastern.edu/en/latest/first_steps/index.html",
"https://rc-docs.northeastern.edu/en/latest/first_steps/passwordlessssh.html",
"https://rc-docs.northeastern.edu/en/latest/first_steps/shellenvironment.html",
"https://rc-docs.northeastern.edu/en/latest/first_steps/usingbash.html",
"https://rc-docs.northeastern.edu/en/latest/hardware/index.html",
"https://rc-docs.northeastern.edu/en/latest/hardware/hardware_overview.html",
"https://rc-docs.northeastern.edu/en/latest/hardware/partitions.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/index.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/introduction.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/accessingood.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/index.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/desktopood.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/fileexplore.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/jupyterlab.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/index.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/understandingqueuing.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/jobscheduling.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/jobscheduling.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/workingwithgpus.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/recurringjobs.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/debuggingjobs.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/index.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/discovery_storage.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/transferringdata.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/globus.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/databackup.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/securityandcompliance.html",
"https://rc-docs.northeastern.edu/en/latest/software/index.html",
"https://rc-docs.northeastern.edu/en/latest/software/systemwide/index.html",
"https://rc-docs.northeastern.edu/en/latest/software/systemwide/modules.html",
"https://rc-docs.northeastern.edu/en/latest/software/systemwide/mpi.html",
"https://rc-docs.northeastern.edu/en/latest/software/systemwide/r.html",
"https://rc-docs.northeastern.edu/en/latest/software/systemwide/matlab.html",
"https://rc-docs.northeastern.edu/en/latest/software/packagemanagers/index.html",
"https://rc-docs.northeastern.edu/en/latest/software/packagemanagers/conda.html",
"https://rc-docs.northeastern.edu/en/latest/software/packagemanagers/spack.html",
"https://rc-docs.northeastern.edu/en/latest/software/fromsource/index.html",
"https://rc-docs.northeastern.edu/en/latest/software/fromsource/makefile.html",
"https://rc-docs.northeastern.edu/en/latest/software/fromsource/cmake.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/index.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/introductiontoslurm.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmcommands.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmrunningjobs.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmmonitoringandmanaging.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmscripts.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmarray.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmbestpractices.html",
"https://rc-docs.northeastern.edu/en/latest/classroom/index.html",
"https://rc-docs.northeastern.edu/en/latest/classroom/class_use.html",
"https://rc-docs.northeastern.edu/en/latest/classroom/cps_ood.html",
"https://rc-docs.northeastern.edu/en/latest/classroom/classroomexamples.html",
"https://rc-docs.northeastern.edu/en/latest/best-practices/index.html",
"https://rc-docs.northeastern.edu/en/latest/best-practices/homequota.html",
"https://rc-docs.northeastern.edu/en/latest/best-practices/checkpointing.html",
"https://rc-docs.northeastern.edu/en/latest/best-practices/optimizingperformance.html",
"https://rc-docs.northeastern.edu/en/latest/best-practices/software.html",
"https://rc-docs.northeastern.edu/en/latest/tutorialsandtraining/index.html",
"https://rc-docs.northeastern.edu/en/latest/tutorialsandtraining/canvasandgithub.html",
"https://rc-docs.northeastern.edu/en/latest/faq.html",
"https://rc-docs.northeastern.edu/en/latest/glossary.html",
]





# Model constants

EMB_INSTRUCTOR_XL = "hkunlp/instructor-xl"
EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"


LLM_FLAN_T5_XXL = "google/flan-t5-xxl"
LLM_FLAN_T5_XL = "google/flan-t5-xl"
LLM_FASTCHAT_T5_XL = "lmsys/fastchat-t5-3b-v1.0"
LLM_FLAN_T5_SMALL = "google/flan-t5-small"
LLM_FLAN_T5_BASE = "google/flan-t5-base"
LLM_FLAN_T5_LARGE = "google/flan-t5-large"

LLM_FALCON_7B_INSTRUCT = "tiiuae/falcon-7b-instruct"
LLM_FALCON_40B_INSTRUCT = "tiiuae/falcon-40b-instruct"
LLM_FALCON_7B = "tiiuae/falcon-7b"
LLM_FALCON_40B = "tiiuae/falcon-40b"

LLM_LLAMA2_70B_INSTRUCT = "upstage/Llama-2-70b-instruct" 
LLM_LLAMA2_7B_32K_INSTRUCT = "togethercomputer/Llama-2-7B-32K-Instruct"#[INST]\n<your instruction here>\n[\INST]\n\n

GPT = "GPT"
LOCAL_FALCON_40B_INSTRUCT_MODEL = "load_local_finetuned_falcon40binstruct_model"

CACHE_DIR = '/work/LitArt/nair/chatbot_files/models'

cache_dir='/work/LitArt/nair/chatbot_files/models'


# Default Config
DEFAULT_CONFIG = {
    "persist_directory": None,
    "load_in_8bit": False,
    "llm": LLM_FALCON_7B_INSTRUCT,
    "quantization" : False,
    "rephrase" : False,
}



# Configs for reranker.py
RERANKER_CONFIG = {
    'score_threshold': 0.1,
    'k': 3, 
    'fetch_k': 15

}

# Configs for chain.py
CHAIN_CONFIG = {
    'temperature': 0.4
}

# Configs for loader.py
LOADER_CONFIG = {
    'chunk_size': 500,
    'chunk_overlap': 300
}

# Configs for model_utils.py
MODEL_UTILS_CONFIG = {
    'create_falcon_7b_instruct': {
        #'load_in_8bit': True,
        'do_sample': True,  # Whether or not to use sampling; use greedy decoding otherwise.
        'max_new_tokens': 400,  # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        'model_kwargs': {
            'device_map': 'auto',
            'load_in_8bit': False,  # This can be overridden with the outer 'load_in_8bit' key if needed
            #'max_length': 512,
            'temperature': 0.4,
            'max_new_tokens': 400, 
            'torch_dtype': 'torch.bfloat16'
        }
        
    },




    'create_falcon_7b': {
        #'load_in_8bit': True,
        'do_sample': True,  # Whether or not to use sampling; use greedy decoding otherwise.
        'max_new_tokens': 400,  # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        'model_kwargs': {
            'device_map': 'auto',
            'load_in_8bit': DEFAULT_CONFIG["load_in_8bit"],  # This can be overridden with the outer 'load_in_8bit' key if needed
            #'max_length': 512,
            'temperature': 0.4,
            'torch_dtype': 'torch.bfloat16'
        }
    },
    'create_falcon_40b': {
        #'load_in_8bit': True,
        'do_sample': True,  # Whether or not to use sampling; use greedy decoding otherwise.
        'max_new_tokens': 400,  # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        'model_kwargs': {
            'device_map': 'auto',
            'load_in_8bit': DEFAULT_CONFIG["load_in_8bit"],  # This can be overridden with the outer 'load_in_8bit' key if needed
            #'max_length': 512,
            'temperature': 0.4,
            'torch_dtype': 'torch.bfloat16'
        }
    },



    'create_falcon_40b_instruct': {
        #'load_in_8bit': True,
        'do_sample': True,  # Whether or not to use sampling; use greedy decoding otherwise.
        'max_new_tokens': 700,  # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        'model_kwargs': {
            'device_map': 'auto',
            'load_in_8bit': DEFAULT_CONFIG["load_in_8bit"],  # This can be overridden with the outer 'load_in_8bit' key if needed
            #'max_length': 512,
            'temperature': 0.2,
            'torch_dtype': 'torch.bfloat16'
        }
    },
    'create_llama2_70b_instruct': {
        #'load_in_8bit': True,
        'do_sample': True,  # Whether or not to use sampling; use greedy decoding otherwise.
        'max_new_tokens': 400,  # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        'model_kwargs': {
            'device_map': 'auto',
            'load_in_8bit': DEFAULT_CONFIG["load_in_8bit"],  # This can be overridden with the outer 'load_in_8bit' key if needed
            #'max_length': 512,
            'temperature': 0.4,
            'torch_dtype': 'torch.bfloat16'
        }
    },
    
    #LLM_LLAMA2_7B_32K_INSTRUCT
    'create_llama2_7b_32k_instruct': {
        #'load_in_8bit': True,
        'do_sample': True,  # Whether or not to use sampling; use greedy decoding otherwise.
        'max_new_tokens': 400,  # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        'model_kwargs': {
            'device_map': 'auto',
            'load_in_8bit': DEFAULT_CONFIG["load_in_8bit"],  # This can be overridden with the outer 'load_in_8bit' key if needed
            #'max_length': 512,
            'temperature': 0.4,
            'torch_dtype': 'torch.bfloat16'
        }
    },
    'gpt': {
        'temperature': 0.5,
        'verbose': True,  
        'do_sample': True,
        'openai_api_key': OPENAI_API_KEY, 
        'max_new_tokens': 500,
        'model_kwargs': {
            'device_map': 'auto',
            'load_in_8bit': DEFAULT_CONFIG["load_in_8bit"],  # This can be overridden with the outer 'load_in_8bit' key if needed
            #'max_length': 512,
            'temperature': 0.4,
            'torch_dtype': 'torch.bfloat16'
        }
    },
    
}





