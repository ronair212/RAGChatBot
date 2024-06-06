from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationTokenBufferMemory
from langchain import PromptTemplate, HuggingFacePipeline
from model_utils import *
from config import DEFAULT_CONFIG
#from langchain.chains import ConversationalRetrievalChain
from ConversationalRetrievalChain import *



def get_llm_pipeline():
    """
    Fetches the appropriate Large Language Model pipeline based on the configuration.
    
    Returns:
        transformers.Pipeline: The relevant HuggingFace pipeline for the LLM.
    """

    #load_in_8bit = DEFAULT_CONFIG["load_in_8bit"]
    
    if DEFAULT_CONFIG["llm"] == LLM_FALCON_40B:
        return create_falcon_40b()
    
    if DEFAULT_CONFIG["llm"] == LLM_LLAMA2_70B_INSTRUCT:
        return create_llama2_70b_instruct()
    
    if DEFAULT_CONFIG["llm"] == LLM_LLAMA2_7B_32K_INSTRUCT:
        return create_llama2_7b_32k_instruct()

    if DEFAULT_CONFIG["llm"] == LLM_FLAN_T5_BASE:
        llm = create_flan_t5_base()
        

    if DEFAULT_CONFIG["llm"] == LLM_FALCON_40B_INSTRUCT:
        llm = create_falcon_40b_instruct()


    if DEFAULT_CONFIG["llm"] == LLM_FALCON_7B_INSTRUCT:
        llm = create_falcon_7b_instruct()
    
        
    if DEFAULT_CONFIG["llm"] == LLM_FALCON_7B:
        llm = create_falcon_7b()

    if DEFAULT_CONFIG["llm"] == GPT:
        llm = gpt()
        
    if DEFAULT_CONFIG["llm"] == LOCAL_FALCON_40B_INSTRUCT_MODEL:
        return load_local_finetuned_falcon40binstruct_model()

        
    return llm

def get_conversational_retrieval_chain(reranker, question_generator, answer_chain, memory):
    """
    Create and return a ConversationalRetrievalChain.

    Args:
        reranker: The reranker object.
        hf_llm: The HuggingFacePipeline object.
        question_generator: The LLMChain object.
        answer_chain: The LLMChain object for generating answers.
        memory: ConversationTokenBufferMemory object.

    Returns:
        ConversationalRetrievalChain: The assembled Conversational Retrieval Chain.
    """
    return ConversationalRetrievalChain(
        retriever=reranker,
        question_generator=question_generator,
        combine_docs_chain=answer_chain,
        verbose=True,
        memory=memory,
        rephrase_question=False,
        rephrasequestion = False,
    )


