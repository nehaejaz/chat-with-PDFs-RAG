import argparse
from langchain_community.vectorstores.chroma import Chroma
from create_db import get_embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os 
import getpass

load_dotenv()


PROMPT_TEMPLATE = """
Answer the question:{question} based only on the following information: {context}

"""

CHROMA_PATH = "./chroma_neha"

def main():
    #Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Please enter the query text")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)
    
def query_rag(query_text: str):
    #Prepare the DB 
    db = Chroma(
        persist_directory= CHROMA_PATH, embedding_function = get_embedding_function()
    )
    
    #Search the DB
    results = db.similarity_search_with_score(query_text,k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results]) 
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        pipeline_kwargs=dict(
            add_to_git_credential=True

        ),
    )

    chat_model = ChatHuggingFace(llm=llm)
    response_text = chat_model.invoke(query_text).content
    
    sources = [docs.metadata.get("id", None) for docs, _score in results]
    formatted_response = f'Response:{response_text}\nSource:{sources}'
    print(formatted_response) 
    formatted_response = {
        "text": response_text,
        "sources" : sources
    }   
    return formatted_response
    
    
if __name__ == "__main__":
    main()
    
