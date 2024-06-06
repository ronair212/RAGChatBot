from loaders import DataLoader
from model_utils import load_huggingface_embeddings
from langchain.vectorstores import Qdrant

'''
def chromadb(urls):
    from langchain.vectorstores import Chroma
    # Load the data using DataLoader
    loader = DataLoader(urls)
    docs = loader.load_and_split()
    embeddings = load_huggingface_embeddings()
    persist_directory = config["persist_directory"]
    chromadatabase = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
    return chromadatabase
'''
'''

On-premise server deployment

No matter if you choose to launch Qdrant locally with a Docker container, or select a Kubernetes deployment with the official Helm chart, 
the way you're going to connect to such an instance will be identical. You'll need to provide a URL pointing to the service.

url = "<---qdrant url here --->"
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=True,
    collection_name="my_documents",
    # force_recreate=True, Setting force_recreate to True allows to remove the old collection and start from scratch.
)

'''


def qdrant_with_server(urls):
    # Load the data using DataLoader
    loader = DataLoader(urls)
    docs = loader.load_and_split()
    
    # Load embeddings using the provided utility function
    embeddings = load_huggingface_embeddings()
    url = "<---qdrant url here --->"
    Qdrantdatabase = Qdrant.from_documents(
        docs,
        embeddings,
        url=url,
        prefer_grpc=True,
        collection_name="RC_documents",
        # force_recreate=True, Setting force_recreate to True allows to remove the old collection and start from scratch.
    )
    return  Qdrantdatabase

def qdrant_without_server(urls):
    # Load the data using DataLoader
    loader = DataLoader(urls)
    docs = loader.load_and_split()

    # Load embeddings using the provided utility function
    embeddings = load_huggingface_embeddings()

    # Load embeddings using the provided utility function
    Qdrantdatabase = Qdrant.from_documents(
        docs,
        embeddings,
        path="/work/LitArt/nair/chatbot_files/tmp/local_qdrant",
        collection_name="RC_documents",
    )
    return  Qdrantdatabase