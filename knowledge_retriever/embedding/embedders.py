import requests
import json
from time import sleep

from .config import settings

class EmbeddingClient:
    def __init__(self,embedding_model, url, **kwargs):
        self.base_url = url
        self.headers = {'Content-Type': 'application/json'}
        self.model = embedding_model

    def vectorize(self, texts, **kwargs):
        url = f'{self.base_url}/vectorize'
        if isinstance(texts, str):
            texts = [texts]
        data = {
        'texts': texts }
        response = requests.post(url, headers=self.headers, data=json.dumps(data))

        if response.status_code == 200:
            return response.json().get('embeddings')
        else:
            raise Exception(f'Request failed with status {response.status_code}')

class EmbeddingLocal:

    def __init__(self, embedding_model, **kwargs):
        self.embedding_model = embedding_model
        if self.embedding_model != "text-embedding-ada-002":
            from sentence_transformers import SentenceTransformer  
            self.model = SentenceTransformer(self.embedding_model)
    
        
    def vectorize(self, texts, **kwargs):
        if self.embedding_model == "text-embedding-ada-002":
            import openai
            openai.api_type = "azure"
            openai.api_base =settings["embeddings"]["OPENAI_API_URL"]
            openai.api_version = "2022-12-01"
            openai.api_key = settings["embeddings"]["OPENAI_API_KEY"]
            embeddings=[]
            max_retries = 6
            i=0
            for text in texts:

                while True:
                    try:

                        if i>max_retries:
                            raise BrokenPipeError
                            
                        response = openai.Embedding.create(input=text,engine=self.embedding_model)
                        embeddings.append(response["data"][0]["embedding"])                        
                        break
                        
                    except openai.error.RateLimitError:
                        i+=1
                        sleeptime = pow(3,i)
                        print("sleep",sleeptime)
                        sleep(sleeptime)

            return embeddings
        else:
            vectorized_texts = self.model.encode(texts, **kwargs)
            return vectorized_texts.tolist()
    
        

        

class Embedding:
    def __init__(self, mode, **kwargs):
        if mode == "remote":
            
            self.embedder = EmbeddingClient(**kwargs)
        elif mode == "local":
            self.embedder = EmbeddingLocal(**kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        

    def vectorize(self, texts, **kwargs):
        return self.embedder.vectorize(texts, **kwargs)

    
if __name__ == "__main__":

     # Create an Embedding object
    embedder = Embedding(mode="local", embedding_model="text-embedding-ada-002")

    # Call the vectorize function
    texts = ["This is a test sentence.", "This is another test sentence."]
    embeddings = embedder.vectorize(texts)
    test = 0
