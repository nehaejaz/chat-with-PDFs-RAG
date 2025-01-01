from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
from enum import Enum
import asyncio
import argparse
import tempfile
import time
import uuid
import os 


load_dotenv()

unique_id = uuid.uuid4().hex[:6]
timestamp = time.strftime("%Y%m%d_%H%M%S")
CHROMA_PATH = f"{unique_id}_{timestamp}"

class DataType(Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Please enter the data path")
    parser.add_argument("data_type", type=str, choices=['markdown', 'pdf'], help="Data type to process. Choose either 'markdown' or 'pdf'.")
    args = parser.parse_args()
    
    data_path = args.data_path
    data_type = args.data_type
    generate_data_store(data_path, data_type)
    
def generate_data_store(files, data_type):
    """
    Description: Loads data from the data diroctary to convert them
    to chunks and store them to the chroma DB.
    """
    documents = []
    if data_type == DataType.PDF.value:
        print("strating pdf extraction...")
        documents = asyncio.run(load_pdf_documents(files))
        
    elif data_type == DataType.MARKDOWN.value:
        documents = load_markdown_documents(files)
    
    if len(documents) != 0:    
        print("lwngth is ",len(documents))
        chunks = split_text(documents)
        save_to_chroma(chunks)
        print("Finished")

    
async def load_pdf_documents(files):
    """
    Description: Reads the streamlit file and creates a temporary PDF
    file for PDF extraction process and retruns them as a list of documents.
    
    Args:
    files (streamlit.runtime.uploaded_file_manager.UploadedFile): Streamlit File Object
    
    Returns:
    documents (List): List of Documents from data dictory 
    """
    documents = []
    print("Extracting Data from PDFs")
    for file in files:
        try:
            if hasattr(file,"read"):
                with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
            else:
                loader = PyPDFLoader(file)
            async for page in loader.alazy_load():
                documents.append(page)
        except Exception as e:
            print(f"Error loading file {file}: {e}")        
    return documents
    
def load_markdown_documents(data_path):
    """
    Description: Reads through the markdown files in the 
    data directory and returns them as a list of documents
    """
    
    documents=[]
    # Load the markdown file using UnstructuredMarkdownLoader
    for filename in os.listdir(data_path):
        loader = UnstructuredMarkdownLoader(os.path.join(DATA_PATH, filename))
        documents.extend(loader.load())
    return documents

def split_text(documents: list[Document]):
    """
    Description: Perfoms Recursive Character chunking on the documents after each new line (\n)
    
    Args:
    documents (List): List of documents to perform chunking on
    
    Returns:
    Chunks (List): Smaller chunks of the document 
    """
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    """
    Description: Saves the chunk to Chroma DB (Vector Database)
    
    Args:
    chunks (List): List of smaller chunks
    """    
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


