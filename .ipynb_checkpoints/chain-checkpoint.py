from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationTokenBufferMemory
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.chains import ConversationalRetrievalChain
#from ConversationalRetrievalChain import *

#from model_utils import create_llama2_70b_instruct, create_falcon_40b, load_huggingface_embeddings
#from config import DEFAULT_CONFIG
from model_utils import *
#from langchain.schema.vectorstore import VectorStoreRetriever
#from langchain.retrievers.document_compressors import CohereRerank
#from langchain_community.vectorstores import Qdrant
#from langchain.retrievers import ContextualCompressionRetriever
from langchain import HuggingFacePipeline
from prompts import *

from config import CHAIN_CONFIG

from loaders import DataLoader
from model_utils import load_huggingface_embeddings

from config import *


# Imports related to reranking
from reranker import *

# Imports related to the conversational retrieval chain creation
from retrieval_chain import *




def initialize_chain():
    """
    Initializes and returns the main conversational retrieval chain.
    
    Returns:
        ConversationalRetrievalChain: The initialized chain.
    """
    # Create LLM pipeline
    llm = get_llm_pipeline()
    
    
    
    #for GPT 
    if DEFAULT_CONFIG['llm'] == "GPT" :
        # Load the large language model
        temperature = CHAIN_CONFIG['temperature']

        # Setup for question generation
        prompt_template = PromptTemplate(
            input_variables=["chat_history", "question"],
            template = question_summarizer_prompt,
        )
        memory = ConversationTokenBufferMemory(llm=llm,memory_key="chat_history", return_messages=True,input_key='question',max_token_limit=500)
        
        system_message_prompt = SystemMessagePromptTemplate(prompt=prompt_template)

        chat_prompt_for_ques = ChatPromptTemplate.from_messages([system_message_prompt])

        question_generator = LLMChain(llm=llm, prompt=chat_prompt_for_ques, verbose=True)

        chat_prompt = PromptTemplate(template=answer_generator_prompt, input_variables=["question", "summaries", "chat_history"])

        answer_chain = load_qa_with_sources_chain(llm, chain_type="stuff", verbose=True, prompt=chat_prompt)
        
        

        reranker = initialize_reranker(DATA_URLS)
        chain = ConversationalRetrievalChain(
                        retriever=reranker,
                        question_generator=question_generator,
                        combine_docs_chain=answer_chain,
                        verbose=True,
                        memory=memory,
                        rephrase_question=False,
                        rephrasequestion = False,
        )
        
        
        return chain
        
        
        
        
    # Load the large language model
    temperature = CHAIN_CONFIG['temperature']
    hf_llm = HuggingFacePipeline(pipeline=llm, model_kwargs={'temperature': temperature})


    # Setup for question generation
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "question"],
        template = question_summarizer_prompt,
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt=prompt_template)

    chat_prompt_for_ques = ChatPromptTemplate.from_messages([system_message_prompt])

    question_generator = LLMChain(llm=hf_llm, prompt=chat_prompt_for_ques, verbose=True)

    chat_prompt = PromptTemplate(template=answer_generator_prompt, input_variables=["question", "summaries", "chat_history"])

    answer_chain = load_qa_with_sources_chain(hf_llm, chain_type="stuff", verbose=True, prompt=chat_prompt)

    memory = ConversationTokenBufferMemory(llm=hf_llm, memory_key="chat_history", return_messages=True, input_key='question', max_token_limit=500)
    
    reranker = initialize_reranker(DATA_URLS)
    chain = get_conversational_retrieval_chain(reranker, question_generator, answer_chain, memory)

    return chain


