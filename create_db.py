from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import openai
from dotenv import load_dotenv
import os 
import shutil
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


load_dotenv()


DATA_PATH = "data/books"
CHROMA_PATH = "chroma_alice"

def main():
    generate_data_store()
    
def generate_data_store():
    documents = load_documents(DATA_PATH)
    chunks = split_text(documents)
    save_to_chroma(chunks)
    
#This function is used to load all the MD files in data/books folder
def load_documents(data_path):
    documents=[]
    # Load the markdown file using UnstructuredMarkdownLoader
    for filename in os.listdir(data_path):
        loader = UnstructuredMarkdownLoader(os.path.join(DATA_PATH, filename))
        documents.extend(loader.load())
    return documents

#This function slpits the documents into smaller chunks for better search 
def split_text(documents: list[Document]):
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
        
    db = Chroma(
        embedding_function=get_embedding_function(), persist_directory=CHROMA_PATH
    )
    chunks_with_id = calculate_chunks(chunks)
    
    #Add/update Documents
    existing_items = db.get(include=[])
    existing_ids = set(existing_items['ids'])
    print(f"Number of existing chunk in DB: {len(existing_ids)}")
    
    new_chunks = []
    for chunk in chunks_with_id:
        if chunk.metadata['id'] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"👉 Adding {len(new_chunks)} chunks in DB")
        new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunks(chunks: list[Document]):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks
          
def get_embedding_function():
    hf_model = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=hf_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return embeddings
if __name__ == "__main__":
    main()


