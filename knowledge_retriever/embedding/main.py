from fastapi import FastAPI
from .embedders import EmbeddingLocal
from sklearn.metrics.pairwise import cosine_similarity
from .config import settings
from pydantic import BaseModel
from typing import List

class Body(BaseModel):
    original: str
    hypothesis: str 
    algoOfComparison: str



class VectorizeRequestBody(BaseModel):
    texts: List[str]



app = FastAPI()

embedder = EmbeddingLocal(**settings["embeddings"])

@app.get("/")
async def root():
    return {"message": "Hello World !"}


@app.post("/score")
async def GetScore(item:Body):
    originalvector = embedder.vectorize([item.original])
    hypothesisvector = embedder.vectorize([item.hypothesis])
    similarityScore = cosine_similarity(originalvector,hypothesisvector)[0][0]
    return {"original":item.original,
            "hypothesis":item.hypothesis,"score":str(similarityScore)}

@app.post("/vectorize")
async def VectorizeText(item:VectorizeRequestBody):
    vectors = embedder.vectorize(item.texts)
    print("\n Embedding Model used: ", embedder.embedding_model)
    return {"embeddings":vectors}
