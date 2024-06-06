#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install pinecone-client


# In[2]:


#!pip install langchain


# In[3]:


#!pip install tiktoken


# In[4]:


#!pip install cohere


# In[5]:


#!pip install openai


# In[6]:


#!pip install chromadb


# In[7]:


#import pinecone


# In[8]:


#pinecone.init(api_key="b360318b-4fc8-4580-bf6c-d88959179985",
#              environment="us-west1-gcp-free")


# In[9]:


#pinecone.whoami()


# In[10]:


#pinecone.list_indexes()


# In[11]:


#pinecone.list_indexes()


# In[12]:


import os
import getpass

os.environ["COHERE_API_KEY"] = getpass.getpass("Cohere API Key:")


# In[13]:


'''
CohereAPIError: You are using a Trial key, which is limited to 2 API calls / minute.
You can continue to use the Trial key for free or upgrade to a Production key with higher rate limits at
'https://dashboard.cohere.ai/api-keys'. 
Contact us on 'https://discord.gg/XW44jPfYJu' or email us at support@cohere.com with any questions
'''


# In[14]:


os.environ["OPENAI_API_KEY"] = getpass.getpass("OPENAI API Key:")


# In[15]:


import langchain


# In[16]:


from langchain.document_loaders import WebBaseLoader


urls = ["https://rc-docs.northeastern.edu/en/latest/runningjobs/understandingqueuing.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/jobscheduling.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/interactiveandbatch.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/workingwithgpus.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/recurringjobs.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/debuggingjobs.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/../datamanagement/index.html",
]
loader = WebBaseLoader(urls)
data = loader.load()


# In[17]:


import tiktoken
encoding_name = tiktoken.get_encoding("cl100k_base")
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# In[18]:


from langchain.text_splitter import TokenTextSplitter
text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=25)
docs = text_splitter.split_documents(data)

'''
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap  = 70,
    length_function = len,
    add_start_index = True,
)
docs = text_splitter.create_documents([data])

for idx, text in enumerate(docs):
    docs[idx].metadata['source'] = "RCDocs"
'''


# In[19]:


type(docs[0])


# In[20]:


docs[0]


# In[21]:


#from langchain.vectorstores import Pinecone
#import pinecone
from langchain.embeddings import CohereEmbeddings


# In[22]:


embeddings = CohereEmbeddings(model='embed-english-light-v2.0',cohere_api_key=os.environ.get("COHERE_API_KEY"))


# In[23]:


#pinecone.init(
#	api_key='b360318b-4fc8-4580-bf6c-d88959179985',
#	environment='us-west1-gcp-free'
#)


# In[24]:


#pinecone.delete_index("chatbot1")


# In[25]:


#pinecone.create_index("chatbot1", dimension=1024)


# In[26]:


#index = pinecone.Index('chatbot1')

#index_name = "chatbot1"


# In[27]:


#docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)


# In[28]:


from langchain.vectorstores import Chroma


# In[29]:


'''
#You can configure Chroma to save and load from your local machine. Data will be persisted automatically and loaded on start (if it exists).

import chromadb

from chromadb.config import Settings


client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", 
                                    persist_directory="db/"
                                ))
#DuckDB on the backend
#stored in db folder
'''


# In[30]:


db = Chroma.from_documents(docs, embeddings)


# In[31]:


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
#from langchain.vectorstores import Pinecone


# In[32]:


# load index
#docsearch = Pinecone.from_existing_index(index_name, embeddings)


# In[33]:


# initialize base retriever
#retriever = docsearch.as_retriever(search_kwargs={"k": 4})
#retriever = Chroma.as_retriever(search_kwargs={"k": 4})
#retriever = Chroma.as_retriever(
#    search_type="mmr",
#    search_kwargs={'k': 4, 'fetch_k': 50} )

from langchain.schema.vectorstore import VectorStoreRetriever
retriever = VectorStoreRetriever(vectorstore=db, search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 10},)

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


# In[34]:


compressor = CohereRerank()


# In[35]:


# Set up cohere's reranker
reranker = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


# In[36]:


from langchain.memory import ConversationTokenBufferMemory


# In[37]:


from langchain.chat_models import ChatOpenAI


# In[38]:


#from langchain.callbacks import ContextCallbackHandler
#from langchain.callbacks import FlyteCallbackHandler
from langchain.callbacks import StdOutCallbackHandler


# In[39]:


#context_callback = ContextCallbackHandler(token="T1gM1n4RzGWLFSsJnQ5ziLUW")
#context_callback = FlyteCallbackHandler()
context_callback = StdOutCallbackHandler()


# In[40]:


llm = ChatOpenAI(temperature=0.7, verbose=True, openai_api_key = os.environ.get("OPENAI_API_KEY"), streaming=True, callbacks=[context_callback])


# In[41]:


memory = ConversationTokenBufferMemory(llm=llm,memory_key="chat_history", return_messages=True,input_key='question',max_token_limit=1000)


# In[42]:


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


system_message_prompt = SystemMessagePromptTemplate(prompt=PromptTemplates)

chat_prompt_for_ques = ChatPromptTemplate.from_messages(
    [system_message_prompt])


# In[43]:


from langchain.chains import LLMChain


# In[44]:


question_generator = LLMChain(llm=llm, prompt=chat_prompt_for_ques, verbose=True)


# In[45]:


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


# In[46]:


from langchain.chains.qa_with_sources import load_qa_with_sources_chain

chat_prompt = PromptTemplate(template=Answer_Generator_Prompt, input_variables=["question", "summaries","chat_history"])

answer_chain = load_qa_with_sources_chain(llm, chain_type="stuff", verbose=True,prompt=chat_prompt)


# In[47]:


from langchain.chains import ConversationalRetrievalChain


chain = ConversationalRetrievalChain(
            retriever=reranker,
            question_generator=question_generator,
            combine_docs_chain=answer_chain,
            verbose=True,
            memory=memory,
            rephrase_question=False
)


# In[48]:


query = "What is the Scheduling Policies for HPC cluster?"
result = chain({"question": query})


print("Question from user : " , query ,"\n")
print("Reply from ChatBot : " , result['answer'])



# In[51]:


query = "How do I check Job Status?"
result = chain({"question": query})


print("Question from user : " , query ,"\n")
print("Reply from ChatBot : " , result['answer'])



# In[ ]:


type(memory)


# In[ ]:


memory


# In[ ]:


'''
Conversation Memory Buffer & Input Details:
------------------------------------------
- Input Key: question
- Memory Key: chat_history
- Maximum Token Limit: 1000
- Return Messages: True

ChatOpenAI Configuration:
-------------------------
- Verbose: True
- Streaming: True
- OpenAI API Key: [REDACTED]
- OpenAI API Base: (No Value Provided)
- OpenAI Organization: (No Value Provided)
- OpenAI Proxy: (No Value Provided)
- Client: ChatCompletion
- Callbacks: FlyteCallbackHandler (Object Reference: 0x7832be39c9d0)

Conversation History:
---------------------
1. User: What is Globus at Northeastern?

   Assistant: Globus is a data management system that allows you to transfer and share files.
   Northeastern has a subscription to Globus, and you can set up a Globus account using your Northeastern credentials.
   You can also link your other accounts, whether they are personal or from another institution, with your Globus account.
   To use Globus, you will need to set up an account and install Globus Connect on your local computer.
   After completing these initial setup procedures, you can use the Globus web app to perform file transfers.

   Sources:
   - RCDocs (Using Globus)
   - RCDocs (Globus Account Set Up)
   - RCDocs (To use Globus)

2. User: How do i set up an account with Globus?

   Assistant: Globus is a data management system that allows you to transfer and share files.
   Northeastern has a subscription to Globus, and you can set up an account with Globus using your Northeastern credentials.
   You can also link your other accounts, whether they are personal or from another institution, with your Globus account.
   To set up an account with Globus, follow these steps:
   1. Go to the Globus website.
   2. Click on "Log In".
   3. Select "Northeastern University" from the options under "Use your existing organizational login" and click "Continue".
   4. Enter your Northeastern username and password.
   5. If you don't have a previous Globus account, click "Continue". If you have an existing account, click "Link to an existing account".
   6. Check the agreement checkbox and click "Continue".
   7. Click "Allow" to permit Globus to access your files.
   After setting up your account, you can access the Globus File Manager app.

   Sources:
   - RCDocs (Using Globus)
   - RCDocs (Globus Account Set Up)
'''


# In[ ]:





# In[ ]:




