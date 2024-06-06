#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import getpass
import time  
import torch

os.environ["COHERE_API_KEY"] = "mVc5EhnC3eDPNFwV3d7ai8IudTmsfijju7h7oxQP"


# In[2]:


os.environ["OPENAI_API_KEY"] = "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"


# In[3]:


import langchain


# In[4]:


file=open("rcdocs_processed.txt","r")
data = file.read()
file.close()


# In[5]:



chunk_size = 500
chunk_overlap = 300
import re
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Split the data string by URLs
pattern = r'(https://[^\s]+)\s(.*?)($|https://)'
matches = re.findall(pattern, data, re.DOTALL)

# Organize data into a dictionary {url: content}
data_dict = {match[0]: match[1].strip() for match in matches}

# Use RecursiveCharacterTextSplitter to split the content
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap  = chunk_overlap,
    length_function = len,
    add_start_index = True,
)

docs = []

for url, content in data_dict.items():
    docs_new = text_splitter.create_documents([content])
    for doc in docs_new:
        doc.metadata['source'] = url
        docs.append(doc)


# In[6]:


type(docs[0])


# In[7]:


docs[57]


# In[8]:


#from langchain.vectorstores import Pinecone
#import pinecone

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()


# In[9]:


#Local mode, without using the Qdrant server, may also store your vectors on disk so they're persisted between runs.
from langchain.vectorstores import Qdrant

Qdrantdb = Qdrant.from_documents(
    docs,
    embeddings,
    path="/work/rc/projects/chatbot/chatbotrc/notebooks/RAG/tmp/local_qdrant",
    collection_name="RC_documents",
)


# In[23]:


'''
#On-premise server deployment
#No matter if you choose to launch Qdrant locally with a Docker container, or select a Kubernetes deployment with the official Helm chart, the way you're going to connect to such an instance will be identical. You'll need to provide a URL pointing to the service.

url = "<---qdrant url here --->"
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=True,
    collection_name="my_documents",
)
'''



# In[10]:


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
#from langchain.vectorstores import Pinecone


# In[11]:


# initialize base retriever
#retriever = docsearch.as_retriever(search_kwargs={"k": 4})
#retriever = Chroma.as_retriever(search_kwargs={"k": 4})
#retriever = Chroma.as_retriever(
#    search_type="mmr",
#    search_kwargs={'k': 4, 'fetch_k': 50} )

from langchain.schema.vectorstore import VectorStoreRetriever
retriever = VectorStoreRetriever(vectorstore=Qdrantdb, search_type="mmr", search_kwargs={'k': 6, 'fetch_k': 10},)

'''

VectorStoreRetriever

Return VectorStoreRetriever initialized from this VectorStore.

Args:
    search_type (Optional[str]): Defines the type of search that
        the Retriever should perform.
Can be "similarity" (default), "mmr", or
"similarity_score_threshold".
    search_kwargs (Optional[Dict]): Keyword arguments to pass to the
        search function. Can include things like:
            k: Amount of documents to return (Default: 4)
            score_threshold: Minimum relevance threshold
                for similarity_score_threshold
            fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
            lambda_mult: Diversity of results returned by MMR;
                1 for minimum diversity and 0 for maximum. (Default: 0.5)
            filter: Filter by document metadata

Returns:
    VectorStoreRetriever: Retriever class for VectorStore.

Examples:

# Retrieve more documents with higher diversity
# Useful if your dataset has many similar documents
docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 6, 'lambda_mult': 0.25}
)

# Fetch more documents for the MMR algorithm to consider
# But only return the top 5
docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 5, 'fetch_k': 50}
)

# Only retrieve documents that have a relevance score
# Above a certain threshold
docsearch.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.8}
)

# Only get the single most similar document from the dataset
docsearch.as_retriever(search_kwargs={'k': 1})

# Use a filter to only retrieve documents from a specific paper
docsearch.as_retriever(
    search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
)
'''


# In[12]:


compressor = CohereRerank() #LLMChainExtractor,LLMChainFilter,EmbeddingsFilter
# will iterate over the initially returned documents and extract from each only the content that is relevant to the query.


# In[13]:


# Set up cohere's reranker
''' instead of immediately returning retrieved documents as-is, 
you can compress them using the context of the given query, so that only the relevant information is returned. '''
reranker = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


# In[14]:


from langchain.memory import ConversationTokenBufferMemory
from langchain.memory import ConversationSummaryBufferMemory


# In[15]:


from langchain.chat_models import ChatOpenAI


# In[16]:



from langchain.callbacks import StdOutCallbackHandler


# In[17]:



context_callback = StdOutCallbackHandler()


# In[ ]:



'''
#VMware/open-llama-7b-open-instruct
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'VMware/open-llama-7b-open-instruct'


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='sequential')

prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

prompt = 'Explain in simple terms how the attention mechanism of a transformer model works'


inputt = prompt_template.format(instruction= prompt)
input_ids = tokenizer(inputt, return_tensors="pt").input_ids.to("cuda")

output1 = model.generate(input_ids, max_length=512)
input_length = input_ids.shape[1]
output1 = output1[:, input_length:]
output = tokenizer.decode(output1[0])

print(output)
'''

EMB_INSTRUCTOR_XL = "hkunlp/instructor-xl"
EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"


LLM_LLAMA2_7B_INSTRUCT = "VMware/open-llama-7b-open-instruct"


cache_dir='/work/rc/projects/chatbot/models'



config = {"persist_directory":None,
          "load_in_8bit":False,
          "embedding" : EMB_SBERT_MPNET_BASE,
          "llm":LLM_LLAMA2_7B_INSTRUCT,
          }


os.environ['TRANSFORMERS_CACHE'] = '/work/rc/projects/chatbot/models'
#cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/work/rc/projects/chatbot/models'



# In[ ]:




def create_sbert_mpnet():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, cache_folder=cache_dir, model_kwargs={"device": device})


    

def create_llama2_7b_instruct(load_in_8bit=False):
        model_name = "VMware/open-llama-7b-open-instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/work/rc/projects/chatbot/models")
        tokenizer = AutoTokenizer.from_pretrained(model_name , cache_dir="/work/rc/projects/chatbot/models")
        hf_pipeline = pipeline(
                task="text-generation",
                model = model,
                do_sample=True,
                tokenizer = tokenizer,
                #trust_remote_code = True,
                max_new_tokens=300,
                #cache_dir=cache_dir,
                model_kwargs={
                    "device_map": "auto", 
                    "load_in_8bit": load_in_8bit, 
                    "max_length": 512, 
                    "temperature": 0.01,
                    
                    "torch_dtype":torch.bfloat16,
                    }
            )
        return hf_pipeline



if config["embedding"] == EMB_SBERT_MPNET_BASE:
    embedding = create_sbert_mpnet()


# In[17]:


load_in_8bit = config["load_in_8bit"]

if config["llm"] == LLM_LLAMA2_7B_INSTRUCT:
    llm = create_llama2_7b_instruct(load_in_8bit=load_in_8bit)
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


#llm = ChatOpenAI(temperature=0.5, verbose=True, openai_api_key = os.environ.get("OPENAI_API_KEY"), streaming=True, callbacks=[context_callback])


# In[85]:


'''
ConversationTokenBufferMemory keeps a buffer of recent interactions in memory,
and uses token length rather than number of interactions to determine when to flush interactions.
'''
#memory = ConversationTokenBufferMemory(llm=llm,memory_key="chat_history", return_messages=True,input_key='question',max_token_limit=1000)
#memory = ConversationSummaryBufferMemory(llm=llm,memory_key="chat_history", return_messages=True,input_key='question',max_token_limit=300)
hf_llm = HuggingFacePipeline(pipeline=llm)

memory = ConversationTokenBufferMemory(llm=hf_llm,memory_key="chat_history", return_messages=True,input_key='question',max_token_limit=750)


# In[20]:


from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain import PromptTemplate

CONDENSE_QUESTION_PROMPT = '''
Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base.
Generate a search query based on the conversation and the new question.

Chat History:
{chat_history}

Question:
{question}

Search query:
'''


'''
PromptTemplates = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base.
Generate a search query based on the conversation and the new question.

Chat History:
{chat_history}

Question:
{question}

Search query:"""
)
'''
PromptTemplates = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
[GENERATE QUESTION]

***Conversation Summary***
Chat History:
{chat_history}

User's New Question:
{question}

Given the Chat History and a User's New Question, rephrase the new question to be a standalone question. """
)




system_message_prompt = SystemMessagePromptTemplate(prompt=PromptTemplates)

chat_prompt_for_ques = ChatPromptTemplate.from_messages(
    [system_message_prompt])


# In[21]:


from langchain.chains import LLMChain


# In[22]:


question_generator = LLMChain(llm=llm, prompt=chat_prompt_for_ques, verbose=True)


# In[24]:


Answer_Generator_Prompt = '''
Respond to given question with factual information from the provided context and chat history.
- If there's insufficient context, mention you don't have enough information.
- If a clarifying question would help, don't hesitate to ask.
- Always include a "SOURCES" section in the response, unless it's a small-talk.
###Question:\n{question}\n\n###

###Context:\n{summaries}\n\n###

###Chat History:\n{chat_history}\n\n###

Response:
'''


# In[27]:


from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
#chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

chat_prompt = PromptTemplate(template=Answer_Generator_Prompt, input_variables=["question", "summaries","chat_history"])

answer_chain = load_qa_with_sources_chain(llm, chain_type="stuff", verbose=True,prompt=chat_prompt)
#answer_chain = load_qa_chain(llm, chain_type="stuff", verbose=True,prompt=chat_prompt)


# In[53]:


from langchain.chains import ConversationalRetrievalChain


chain = ConversationalRetrievalChain(
            retriever=reranker,
            question_generator=question_generator,
            combine_docs_chain=answer_chain,
            verbose=True,
            memory=memory,
            rephrase_question=True
)


# In[54]:


query = "What is the Scheduling Policies for HPC cluster?"
result = chain({"question": query})

print("\n\n\n\n\n\n")
print("Question from user : " , query ,"\n")
print("Reply from ChatBot : " , result['answer'])



# In[55]:


query = "tell me more about fifo"
result = chain({"question": query})

print("\n\n\n\n\n\n")
print("Question from user : " , query ,"\n")
print("Reply from ChatBot : " , result['answer'])



# In[56]:


query = "tell me about globus"
result = chain({"question": query})

print("\n\n\n\n\n\n")
print("Question from user : " , query ,"\n")
print("Reply from ChatBot : " , result['answer'])




# In[84]:


query = "do you like science?"
result = chain({"question": query})

print("\n\n\n\n\n\n")
print("Question from user : " , query ,"\n")
print("Reply from ChatBot : " , result['answer'])





# In[58]:


query = "is northeastern a good university?"
result = chain({"question": query})

print("\n\n\n\n\n\n")
print("Question from user : " , query ,"\n")
print("Reply from ChatBot : " , result['answer'])





# In[65]:


query = "tell me more about Priority-Based"
result = chain({"question": query})

print("\n\n\n\n\n\n")
print("Question from user : " , query ,"\n")
print("Reply from ChatBot : " , result['answer'])





# In[ ]:





# In[47]:


type(memory)


# In[60]:


memory


# In[ ]:





# In[ ]:





# In[49]:


chain.memory.memory_key


# In[83]:


chain.memory.chat_memory.messages 


# In[63]:


type(chain.memory.chat_memory.messages[0])


# In[48]:


del memory


# In[64]:


chain.memory.chat_memory.messages = chain.memory.chat_memory.messages[0]


# In[ ]:


human_messages = [HumanMessage(content=message) for message in messages]


# In[82]:


chain.memory.chat_memory.messages


# In[72]:


from langchain.schema.messages import HumanMessage


# In[78]:


#human_messages = [HumanMessage(content=message) for message in chain.memory.chat_memory.messages]

human_messages = [
    HumanMessage(content=message.content) 
    for message in chain.memory.chat_memory.messages 
    if isinstance(message, HumanMessage)
]


# In[74]:


from langchain.schema.messages import HumanMessage

# Example list of strings that you want to convert
messages = [
    "tell me about globus",
    "do you like science?"
]

# Convert the list of strings to HumanMessage objects
human_messages = [HumanMessage(content=message) for message in messages]

# Optionally, if you have additional attributes to add you could do:
# human_messages = [HumanMessage(content=message, additional_kwargs={'key': 'value'}) for message in messages]


# In[81]:


human_messages


# In[ ]:





# In[ ]:





# In[51]:


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()

