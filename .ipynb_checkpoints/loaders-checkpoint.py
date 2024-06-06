import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import TokenTextSplitter
from config import *
from config import LOADER_CONFIG

import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

        

class DataLoader:
    def __init__(self, urls, 
                 chunk_size=LOADER_CONFIG['chunk_size'], 
                 chunk_overlap=LOADER_CONFIG['chunk_overlap']):
        """
        Initialize DataLoader with a list of URLs.

        Args:
            urls (list): List of URLs to fetch data from.
            chunk_size (int, optional): Size of each chunk for TokenTextSplitter. Defaults to 500.
            chunk_overlap (int, optional): Overlap size for TokenTextSplitter. Defaults to 25.
        """
        self.urls = urls
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split(self):
        """
        Load data from the URLs and split them into chunks using TokenTextSplitter.

        Returns:
            list: List of splitted documents.
        """
        #loader = WebBaseLoader(self.urls)
        #data = loader.load()
        file=open("rcdocs_processed_new.txt","r")
        data = file.read()
        file.close()        
        
        
        # Split the data string by URLs
        pattern = r'(https://rc-docs\.northeastern\.edu/en/[^ ]+\.html)'
        
        # Using re.split to split the text, keeping the URLs
        split_data = re.split(pattern, data)

        # Initializing an empty dictionary
        data_dict = {}

        # Iterating through the split data to pair URLs with their corresponding text
        for i in range(1, len(split_data), 2):
            url = split_data[i].strip()
            content = split_data[i + 1].strip()
            data_dict[url] = content



        # Use RecursiveCharacterTextSplitter to split the content
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size = self.chunk_size,
            chunk_overlap  = self.chunk_overlap,
        )

        docs_split_for_qdrant = []

        for url, content in data_dict.items():
            docs_new = text_splitter.create_documents([content])
            for doc in docs_new:
                doc.metadata['source'] = url
                docs_split_for_qdrant.append(doc)
                            
                            
                            
        return docs_split_for_qdrant

        


'''

# Split the data string by URLs
        pattern = r'(https://[^\s]+)\s(.*?)($|https://)'
        matches = re.findall(pattern, data, re.DOTALL)

        # Organize data into a dictionary {url: content}
        data_dict = {match[0]: match[1].strip() for match in matches}

        # Use RecursiveCharacterTextSplitter to split the content
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap  = self.chunk_overlap,
            length_function = len,
            add_start_index = True,
        )

        docs_split_for_qdrant = []

        for url, content in data_dict.items():
            docs_new = text_splitter.create_documents([content])
            for doc in docs_new:
                doc.metadata['source'] = url
                docs_split_for_qdrant.append(doc)
                
                
# If you need to use the loader directly from this module:
if __name__ == "__main__":
    from config import DATA_URLS

    data_loader = DataLoader(DATA_URLS)
    docs = data_loader.load_and_split()
    for doc in docs:
        print(doc)
'''