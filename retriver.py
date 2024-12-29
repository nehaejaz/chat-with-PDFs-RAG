"""
Module for retieving conetxt from Data
"""
# from langchain_community.vectorstores.chroma import Chroma
from langchain_chroma import Chroma

class Retriever:
    """
    Attributes:
    query (str) : User's query
    embedding_function : Function used for vectorization
    top_k (int): Number of top chunks to retrieve 
    """
    def __init__(self, CHROMA_PATH:str, query:str, embedding_function, top_k=5, ):
        self.CHROMA_PATH = CHROMA_PATH
        self.query = query
        self.embedding_function = embedding_function
        self.top_k = top_k  
    
    def retrieve_context(self):
        db = Chroma(
            persist_directory= self.CHROMA_PATH, embedding_function = self.embedding_function()
        )
        results = db.similarity_search_with_score(self.query,self.top_k)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results]) 
     
        return context_text
    
    def predict(self):
        return self.retrieve_context()
