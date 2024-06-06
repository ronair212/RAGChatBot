#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import getpass

os.environ["COHERE_API_KEY"] = "mVc5EhnC3eDPNFwV3d7ai8IudTmsfijju7h7oxQP"


# In[2]:


os.environ["OPENAI_API_KEY"] = "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"


# In[3]:


import langchain


# In[4]:


'''
from langchain.document_loaders import WebBaseLoader


urls = [
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/get_access.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/accountmanager.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/connectingtocluster/mac.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/connectingtocluster/windows.html",
"https://rc-docs.northeastern.edu/en/latest/first_steps/passwordlessssh.html",
"https://rc-docs.northeastern.edu/en/latest/first_steps/shellenvironment.html",
"https://rc-docs.northeastern.edu/en/latest/first_steps/usingbash.html",
"https://rc-docs.northeastern.edu/en/latest/hardware/hardware_overview.html",
"https://rc-docs.northeastern.edu/en/latest/hardware/partitions.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/introduction.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/accessingood.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/desktopood.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/fileexplore.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/jupyterlab.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/understandingqueuing.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/jobscheduling.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/jobscheduling.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/workingwithgpus.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/recurringjobs.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/debuggingjobs.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/discovery_storage.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/transferringdata.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/globus.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/databackup.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/securityandcompliance.html",
"https://rc-docs.northeastern.edu/en/latest/software/systemwide/modules.html",
"https://rc-docs.northeastern.edu/en/latest/software/systemwide/mpi.html",
"https://rc-docs.northeastern.edu/en/latest/software/systemwide/r.html",
"https://rc-docs.northeastern.edu/en/latest/software/systemwide/matlab.html",
"https://rc-docs.northeastern.edu/en/latest/software/packagemanagers/conda.html",
"https://rc-docs.northeastern.edu/en/latest/software/packagemanagers/spack.html",
"https://rc-docs.northeastern.edu/en/latest/software/fromsource/makefile.html",
"https://rc-docs.northeastern.edu/en/latest/software/fromsource/cmake.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/introductiontoslurm.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmcommands.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmrunningjobs.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmmonitoringandmanaging.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmscripts.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmarray.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmbestpractices.html",
"https://rc-docs.northeastern.edu/en/latest/classroom/class_use.html",
"https://rc-docs.northeastern.edu/en/latest/classroom/cps_ood.html",
"https://rc-docs.northeastern.edu/en/latest/classroom/classroomexamples.html",
"https://rc-docs.northeastern.edu/en/latest/best-practices/homequota.html",
"https://rc-docs.northeastern.edu/en/latest/best-practices/checkpointing.html",
"https://rc-docs.northeastern.edu/en/latest/best-practices/optimizingperformance.html",
"https://rc-docs.northeastern.edu/en/latest/best-practices/software.html",
"https://rc-docs.northeastern.edu/en/latest/tutorialsandtraining/canvasandgithub.html",
]
loader = WebBaseLoader(urls)
data = loader.load()
'''


# In[5]:


file=open("rcdocs_processed.txt","r")
data = file.read()
file.close()

'''
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
    chunk_size = 700,
    chunk_overlap  = 70,
    length_function = len,
    add_start_index = True,
)

all_docs = []

for url, content in data_dict.items():
    docs = text_splitter.create_documents([content])
    for doc in docs:
        doc.metadata['source'] = url
        all_docs.append(doc)
'''    
        


# In[6]:



from langchain.embeddings import HuggingFaceEmbeddings
#Local mode, without using the Qdrant server, may also store your vectors on disk so they're persisted between runs.
from langchain.vectorstores import Qdrant
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory


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


#from langchain.vectorstores import Pinecone



embeddings = HuggingFaceEmbeddings()


# In[7]:





'''
PromptTemplates = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base.
Generate a question based on the conversation chat history and the new question.

Chat History:
{chat_history}

New Question:
{question}

Generated question:"""
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



# In[8]:



Answer_Generator_Prompt= '''
<Instructions>
Important:
Answer like a technical assistant with the facts from the context and chat history given below. If there isn't enough information in context, say you don't know.
If asking a clarifying question to the user would help, ask the question.
ALWAYS return a "SOURCES" part in your answer, except for small-talk conversations.

Question: {question}
Context:
---------------------
    {summaries}
---------------------

Chat History:
{chat_history}

</Instructions>
'''


# In[9]:


Answer_Generator_Prompt = '''
[TECHNICAL ASSISTANT RESPONSE]

***Instructions***
    - Respond with factual information from the provided context and chat history.
- If there's insufficient context, mention you don't have enough information.
- If a clarifying question would help, don't hesitate to ask.
- Always include a "SOURCES" section in the response, unless it's a small-talk.

***Details***
Question: {question}

Context:
---------------------
{summaries}
---------------------

Chat History:
{chat_history}
'''


# In[10]:


import pandas as pd
import os


chunk_sizes = [400 ,800 , 1000]
chunk_overlaps = [30 , 100 , 200 ]

k_values = [3,5,7]
fetch_k_values = []
lambda_mult_values = [ 0.5 ,0.75  , 0.9]
filter_values = []
search_params_values = []
score_threshold_values= [0.6,0.8,0.9]


input_path = 'temp_sample_questions.xlsx'
df_input = pd.read_excel(input_path)
queries = df_input['queries'].tolist()


# In[12]:


i = 0
rephrase_questions = [True , False ]
for rephrase_question in rephrase_questions:
    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            for k_value in k_values:
                for lambda_mult_value in lambda_mult_values:

                    print("\n\n\n\n\n\n")
                    print("current values are chunk_size = " , chunk_size," chunk_overlap = " ,chunk_overlap ," k_value = ",k_value," lambda_mult_value = " ,lambda_mult_value , " rephrase_question = " , rephrase_question)
                    i = i + 1 

                    #text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    #docs = text_splitter.split_documents(data)

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


                    name  = "RC_documents" + str(i)
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

                    llm = ChatOpenAI(temperature=0.5, verbose=True, openai_api_key = os.environ.get("OPENAI_API_KEY"), streaming=True, callbacks=[context_callback])
                    memory = ConversationSummaryBufferMemory(llm=llm,memory_key="chat_history", return_messages=True,input_key='question',max_token_limit=500)
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
                        rephrase_question=rephrase_question )





                    '''
                    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
                    embeddings = OpenAIEmbeddings()
                    vectordb = Chroma(embedding_function=embeddings, persist_directory=directory)
                    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                    qa = ConversationalRetrievalChain.from_llm(
                        model,
                        vectordb.as_retriever(),
                        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                        memory=memory
                    )
                    '''

                    # Iterate over queries and save results
                    answers = []
                    for query in queries:
                        print("\n\n\n")
                        print("running query " , query)
                        print(f'output_chunk_size_{chunk_size}_overlap_{chunk_overlap}_k_value_{k_value}_lambda_mult_value_{lambda_mult_value}_rephrase_question_{rephrase_question}')
                        result = chain({"question": query})
                        print("answer is " , result['answer'])
                        #answers.append(result['answer'])
                        time.sleep(60)

                    df_output = pd.DataFrame({'queries': queries, 'answers': answers})

                    # Save the output to an Excel sheet
                    output_filename = f'output_chunk_size_{chunk_size}_overlap_{chunk_overlap}_k_value_{k_value}_lambda_mult_value_{lambda_mult_value}_{rephrase_question}.xlsx'
                    df_output.to_excel(output_filename, index=False)
                    del memory
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




