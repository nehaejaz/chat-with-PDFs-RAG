"""
Module for generating LLM response 
"""
import cohere
import os 

class ResponseGenerator:
    """
    Args:
    cohere_client: cohere.ClientV2
    model: str
    """
    
    def __init__(self, model):
        self.cohere_client: cohere.ClientV2 = cohere.ClientV2(
            api_key=os.environ["COHERE_API_KEY"]
        )
        self.model: str = model
    
    def generate_response(self, query, prompt, context):
        messages= [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]
        
        response = self.cohere_client.chat(
            messages = messages,
            model = self.model,
            max_tokens=200,
            temperature = 0.1,
            # documents= context
        )
        
        return response.message.content[0].text
    
    def predict(self, query, prompt, context):
        return self.generate_response(query, prompt, context)