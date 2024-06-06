from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema.vectorstore import VectorStoreRetriever
# Imports related to database management
from database_manager import qdrant_with_server, qdrant_without_server
from config import *



import os
import getpass

os.environ["COHERE_API_KEY"] = "mVc5EhnC3eDPNFwV3d7ai8IudTmsfijju7h7oxQP"


# In[2]:


def initialize_reranker(urls):
    #Qdrantdb = qdrant_with_server(urls)
    Qdrantdb = qdrant_without_server(urls)

    retriever = VectorStoreRetriever(
        vectorstore=Qdrantdb, 
        search_type="similarity", 
        search_kwargs={'k': RERANKER_CONFIG['k'],
                       #'fetch_k': RERANKER_CONFIG['fetch_k'] 
                       #, 'score_threshold' : RERANKER_CONFIG['score_threshold']
                      } ,
        )
    
    compressor = CohereRerank(top_n = 3)

    reranker = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=retriever
        )

    return reranker



