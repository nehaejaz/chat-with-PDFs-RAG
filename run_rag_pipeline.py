import argparse
from create_db import get_embedding_function
from rag_pipeline import RAGPipeline
from response_generator import ResponseGenerator
from retriver import Retriever
from langchain import hub


PROMPT_TEMPLATE = hub.pull("rlm/rag-prompt")

# PROMPT_TEMPLATE = """
# Answer the question:{question} based only on the following information: {context}

# """
# prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)

def main():
    #Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Please enter the query text")
    parser.add_argument("schema_name", type=str, help="Please enter the schema name")
    args = parser.parse_args()
    query = args.query_text
    schema_name = args.schema_name
    response = run_rag_pipeline(schema_name,query)
    print(response)
    
def run_rag_pipeline(schema_name, query: str):
    retriever = Retriever(schema_name, query, get_embedding_function)
    response_generator = ResponseGenerator(model="command-r")
    rag_pipeline = RAGPipeline(
        PROMPT_TEMPLATE, 
        retiever=retriever, 
        response_generator=response_generator
    )
    response = rag_pipeline.predict(query)
      
    return response
    
    
if __name__ == "__main__":
    main()
    
