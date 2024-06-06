#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import getpass

os.environ["COHERE_API_KEY"] = "Aj7fRPV0FBm1u6baUBuAZc5yMOvOs6krkrqVppam"



# In[3]:


os.environ["OPENAI_API_KEY"] = "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"



# In[4]:


import langchain


# In[5]:


from langchain.document_loaders import WebBaseLoader


urls = ["https://rc-docs.northeastern.edu/en/latest/welcome/index.html",
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
loader = WebBaseLoader(urls)
data = loader.load()


# In[6]:



from langchain.embeddings import HuggingFaceEmbeddings
#Local mode, without using the Qdrant server, may also store your vectors on disk so they're persisted between runs.
from langchain.vectorstores import Qdrant
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank


from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.memory import ConversationTokenBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import TokenTextSplitter
import time




embeddings = HuggingFaceEmbeddings()


# In[7]:



CONDENSE_QUESTION_PROMPT = '''
Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base.
Generate a search query based on the conversation and the new question.

Chat History:
{chat_history}

Question:
{question}

Search query:
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



# In[8]:


Answer_Generator_Prompt= '''
<Instructions>
Important:
Answer with the facts listed in the list of sources below. If there isn't enough information below, say you don't know.
If asking a clarifying question to the user would help, ask the question.
ALWAYS return a "SOURCES" part in your answer, except for small-talk conversations.

Question: {question}
Sources:
---------------------
    {summaries}
---------------------

Chat History:
{chat_history}
'''


# In[9]:


import pandas as pd
import os


chunk_sizes = [200,400,800]
chunk_overlaps = [40,100]
#chunk_sizes = [400,800]
#chunk_overlaps = [40, 80]
k_values = [6,8]
fetch_k_values = []
lambda_mult_values = [0.5 , 0.75 , 0.9]
filter_values = []
search_params_values = []
score_threshold_values = [0.6,0.8,0.9]


input_path = 'sample_questions.xlsx'
df_input = pd.read_excel(input_path)
queries = df_input['queries'].tolist()


# In[ ]:


i = 0

for chunk_size in chunk_sizes:
    for chunk_overlap in chunk_overlaps:
        for k_value in k_values:
            for lambda_mult_value in lambda_mult_values:
                
            
                print("current values are chunk_size = " , chunk_size," chunk_overlap = " ,chunk_overlap ," k_value = ",k_value," lambda_mult_value = " ,lambda_mult_value)
                i = i + 1 

                text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                docs = text_splitter.split_documents(data)
                name  = "RC_documents_Copy1" + str(i)
                Qdrantdb = Qdrant.from_documents(
                    docs,
                    embeddings,
                    path="/work/rc/projects/chatbot/chatbotrc/notebooks/RAG/tmp/local_qdrant"+name,
                    collection_name=name ,
                )
                print("Qdrantdb with " , name , " completed" )


                retriever = VectorStoreRetriever(vectorstore=Qdrantdb, search_type="mmr", search_kwargs={'k': k_value, 'fetch_k': 10 , 'lambda_mult' : lambda_mult_value},)
                compressor = CohereRerank()

                reranker = ContextualCompressionRetriever(
                    base_compressor=compressor, base_retriever=retriever)

                context_callback = StdOutCallbackHandler()

                llm = ChatOpenAI(temperature=0.7, verbose=True, openai_api_key = os.environ.get("OPENAI_API_KEY"), streaming=True, callbacks=[context_callback])
                memory = ConversationTokenBufferMemory(llm=llm,memory_key="chat_history", return_messages=True,input_key='question',max_token_limit=1000)
                system_message_prompt = SystemMessagePromptTemplate(prompt=PromptTemplates)

                chat_prompt_for_ques = ChatPromptTemplate.from_messages([system_message_prompt])
                question_generator = LLMChain(llm=llm, prompt=chat_prompt_for_ques, verbose=True)
                chat_prompt = PromptTemplate(template=Answer_Generator_Prompt, input_variables=["question", "summaries","chat_history"])
                answer_chain = load_qa_with_sources_chain(llm, chain_type="stuff", verbose=True,prompt=chat_prompt)
                chain = ConversationalRetrievalChain(
                    retriever=reranker,
                    question_generator=question_generator,
                    combine_docs_chain=answer_chain,
                    verbose=True,
                    memory=memory,
                    rephrase_question=False )

                # Iterate over queries and save results
                answers = []
                for query in queries:
                    print("running query " , query)
                    result = chain({"question": query})
                    print("answer is " , result['answer'])
                    answers.append(result['answer'])
                    time.sleep(60)

                df_output = pd.DataFrame({'queries': queries, 'answers': answers})

                # Save the output to an Excel sheet
                output_filename = f'output_chunk_size_{chunk_size}_overlap_{chunk_overlap}_k_value_{k_value}_lambda_mult_value_{lambda_mult_value}_Copy1.xlsx'
                df_output.to_excel(output_filename, index=False)

print("Finished processing all combinations!")


# In[ ]:





# In[ ]:


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
_values
k_values = [2,4,6,8]
fetch_k_values = []
lambda_mult_values = [0.5 , 0.75 , 0.9]
filter_values = []
search_params_values = []
score_threshold_values = [0.6,0.8,0.9]

#retriever = VectorStoreRetriever(vectorstore=Qdrantdb, search_type="mmr", search_kwargs={'k': k_values, 'fetch_k': fetch_k_values},)

param k: Number of Documents to return. Defaults to 4. :
param fetch_k: Number of Documents to fetch to pass to MMR algorithm. Defaults to 20. :
param lambda_mult: Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5. :
param filter: Filter by metadata. Defaults to None. :
param search_params: Additional search params :
param score_threshold: Define a minimal score threshold for the result. If defined, less similar results will not be returned. 

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


# In[ ]:




