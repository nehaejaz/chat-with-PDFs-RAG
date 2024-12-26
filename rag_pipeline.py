from langchain.prompts import ChatPromptTemplate

class RAGPipeline:
    """
    Attributes:
    PROMPT_TEMPLATE: The template of the prompt passed to LLM for answering the question
    retiever: Module that rertieves the context based on query
    response_generator: Module that generates the response based using LLM
    """
    def __init__(self, PROMPT_TEMPLATE, retiever, response_generator):
        self.PROMPT_TEMPLATE = PROMPT_TEMPLATE
        self.response_generator = response_generator
        self.retriever = retiever 
    
    def predict(self, query: str):
        
        context = self.retriever.predict()
        prompt = self.PROMPT_TEMPLATE.format(context=context, question=query)
        response = self.response_generator.predict(query,prompt, context)
       
        # sources = [docs.metadata.get("id", None) for docs, _score in results]
        
        formatted_response = {
            "text": response,
            # "sources" : sources
        }

        return formatted_response
        