from typing import Union, Annotated, List
import io

from fastapi import FastAPI, File, Form, UploadFile,HTTPException
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

from .config import settings
from .vectorstore.documentretriever import KnowledgeRetriever

from .vectorstore.chunksplitter import ChunkMethod
import logging

logging.basicConfig(level=logging.DEBUG)
SUPPORTED_FILES = (".eml", ".html", ".json", ".md", ".msg", ".rst", ".rtf", ".txt", ".xml", ".jpeg", ".png", ".csv", ".doc", ".docx", ".epub", ".odt", ".pdf", ".ppt", ".pptx", ".tsv", ".xlsx")

app = FastAPI()

print("settings", settings.to_dict())
retriever = KnowledgeRetriever(**settings["vectorstore"])

class QueryBody(BaseModel):
    Texts: List[str]
    N_results: int
    Collection_name: str

# Base model with common fields
class BaseChunkBody(BaseModel):
    collection_name: str
    chunk_ids: List[str]


# Extended model where 'chunks text' are required
class UpdateBodyWithChunks(BaseChunkBody):
    chunks: List[str]



@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/ping")
async def root():
    return {"message": "Hello World !"}

@app.post("/query")
async def get_documents(item: QueryBody):
    if isinstance(item.Texts, str):
        item.Texts = [item.Texts]
    documents = retriever.retrieve_documents(
        user_texts=item.Texts,
        collection_name=item.Collection_name,
        n_results=item.N_results,
        **settings["embeddings"]
    )
    logging.info(f'This is the documents in the  Knowledge Retriever itself {documents}')

    return documents


@app.post("/add_document")
async def add_document(
    collection_name: Annotated[str, Form()],
    file: UploadFile,
    chunk_size: Annotated[Union[int, None], Form()] = 700,
    chunk_overlap: Annotated[Union[int, None], Form()] = 140,
    chunk_method_name: Annotated[ChunkMethod, Form()] = ChunkMethod.method3
):
    
    
    chunkmethod = ChunkMethod(chunk_method_name)

    content = await file.read()
    file_name = file.filename
    file = io.BytesIO(content)
    if isValidFormat(file_name):
        chunk_strings = retriever.add_file_to_store(
            file,
            collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_method_name=chunkmethod,
            file_filename=file_name
        )
        print("Processed chunk : ", chunk_strings[0].metadata)
        return chunk_strings
    else:
        raise HTTPException(status_code=415, detail=f"Stopped at {file_name}")



@app.post("/add_multi_documents")
async def add_multi_documents(
    collection_name: Annotated[str, Form()],
    files: List[UploadFile],
    chunk_size: Annotated[Union[int, None], Form()] = 700,
    chunk_overlap: Annotated[Union[int, None], Form()] = 140,
    chunk_method_name: Annotated[ChunkMethod, Form()] = ChunkMethod.method3
):
    
    chunkmethod = ChunkMethod(chunk_method_name)

    all_chunk_strings = []
    for file in files:
        file_name = file.filename  # Store the filename in a variable
        if isValidFormat(file_name):
            print(">>>> FileName: >>>>", file_name)
            contents = await file.read()
            file = io.BytesIO(contents)
            chunk_strings = retriever.add_file_to_store(
                file,
                collection_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_method_name=chunkmethod,
                file_filename=file_name
            )
            all_chunk_strings.extend(chunk_strings)
            if chunk_strings:
                print(f"Processed file: {file_name}, chunk: {chunk_strings[0].metadata}")
       
        else:
            raise HTTPException(status_code=415, detail=f"Stopped at {file_name}")
    return all_chunk_strings



def isValidFormat(filename):

    if "."+filename.split(".")[-1] in SUPPORTED_FILES:
        return True
    else :
        return False        


@app.get("/list_collections")
async def list_collections():
    return retriever.list_collections()


@app.post("/update_chunks")
async def update_chunks(item: UpdateBodyWithChunks):
    return retriever.update_chunks(collection_name=item.collection_name, chunks_ids=item.chunk_ids, chunks=item.chunks)

@app.post("/delete_chunks")
async def delete_chunks(item: BaseChunkBody):
    return retriever.delete_chunks(collection_name=item.collection_name,chunks_ids=item.chunk_ids)


@app.post("/get_chunks_by_id")
async def get_chunks(item: BaseChunkBody):
    return retriever.get_chunks(collection_name=item.collection_name,chunks_ids=item.chunks_ids)

@app.post("/collection_peek")
async def collection_peek(collection_name: str = Form(...)):
    return retriever.list_chunks(collection_name=collection_name)

@app.post("/delete_collection")
async def delete_collection(collection_name: str = Form(...)):
    return retriever.delete_collection(collection_name=collection_name)