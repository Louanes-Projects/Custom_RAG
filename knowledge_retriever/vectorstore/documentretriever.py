import copy
import json
import requests
from .chunksplitter import ChunkSplitter
from pydantic import BaseModel, Field
from .config import settings
from ..embedding.embedders import Embedding
import uuid
import logging

logging.basicConfig(level=logging.DEBUG)

PDF_PARSING_STRATEGY = "fast"
PDF_COLLATING_MODE = "single"


class Document(BaseModel):
    """Interface for interacting with a document."""

    page_content: str
    metadata: dict = Field(default_factory=dict)


class KnowledgeRetrieverLocal:
    def __init__(self, **kwargs):
        from chromadb import Client
        from chromadb.config import Settings
        from unstructured.partition.auto import partition
        self.partition = partition
        self.embedder = Embedding(**settings["embeddings"])
        self.embedding_function = self.embedder.vectorize
        chroma_settings = Settings(
            chroma_db_impl="chromadb.db.duckdb.PersistentDuckDB",
            persist_directory=settings["vectorstore"]["persist_directory"],
        )
        self.vectorstore = Client(chroma_settings)
        self._default_collection_name = "default_collection"
        # WARNING self.collection is deprecated
        self.collection = self.vectorstore.get_or_create_collection(
            self._default_collection_name, embedding_function=self.embedding_function
        )

    def retrieve_documents(self, user_texts, collection_name=None, n_results=1, **kwargs):
        """Return most similar chunks for each `text` in `user_texts`"""
        if collection_name is None:
            collection_name = self._default_collection_name
        logging.info(f'collection ID used in KR {collection_name}')

        query_embeddings = self.embedder.vectorize(user_texts)
        if isinstance(query_embeddings[0], float):
            query_embeddings = [query_embeddings]

        collection = self.vectorstore.get_or_create_collection(
            collection_name, embedding_function=self.embedding_function
        )
        logging.info(f'collection used in KR {collection}')

        docs_and_scores = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["embeddings", "documents", "distances", "metadatas"],
        )

        return self.mapper_chroma2langchain(docs_and_scores)

    def add(self, chunks,collection_id) -> None:
        collection = self.vectorstore.get_or_create_collection(collection_id,embedding_function=self.embedding_function)
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [metadata['chunk_id'] for metadata in metadatas]
        collection.add(documents=texts, embeddings=None, metadatas=metadatas, ids=ids)
  


    def add_file_to_store(
        self,
        file: bytes,
        collection_id,
        chunk_method_name,
        chunk_size=500,
        chunk_overlap=140,
        file_filename=None,
    ):
        elements = []
        try:
            elements = self.partition(file=file, file_filename=file_filename, strategy=PDF_PARSING_STRATEGY)
        except UnicodeDecodeError as e:
            print("Warning from partition", (file_filename, e))
        except IndexError as e:
            print("Warning from partition", (file_filename, e))
        except KeyError as e:
            print("Warning from partition", (file_filename, e))
            
        if len(elements) == 0:
            return []
        documents = self._mapper_unstructured2langchain(
            elements, mode=PDF_COLLATING_MODE
        )

        #split text
        chunksplitter = ChunkSplitter(documents=documents)
        
        if chunk_method_name == "clustering_adjacent_sentences":
            params = {"min_chars": 60, "max_chars":3000, "pipeline": "fr_core_news_sm"} 
        else:
            params = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}  # These should be the parameters provided by the user            

        chunks = chunksplitter.split_text(chunk_method_name, **params)
        
        chunk_ids = self.generate_uuids(chunks)

        for index,chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = chunk_ids[index]
            chunk.metadata["filename"] = file_filename
            chunk.metadata.pop("file_directory",None)
            chunk.metadata.pop("header_footer_type",None)
            chunk.metadata.pop("page_number",None)
            chunk.metadata.pop("text_as_html",None)
            chunk.metadata["chunk_method"] = chunk_method_name
            chunk.metadata["length"] = len(chunk.page_content)

        try:
            self.add(chunks,collection_id)
        except BrokenPipeError as e:    
            #TODO use logger minio handler 
            print(f"Didn't vectorize this file {file_filename}")

        return chunks



    

    def list_collections(self):
        collections = self.vectorstore.list_collections()
        return collections

    def update_chunks(self,collection_id, chunk_ids :[], chunks):
        collection = self.vectorstore.get_collection(collection_id,embedding_function=self.embedding_function)
        collection.update(ids=chunk_ids,documents=chunks)
        return self.get_chunks(collection_id,chunk_ids)
    
    def get_chunks(self,collection_id,chunk_ids):
        collection = self.vectorstore.get_collection(collection_id,embedding_function=self.embedding_function)
        chunks = collection.get(ids=chunk_ids, include=["embeddings", "documents", "metadatas"])
        return chunks
    
    def delete_chunks(self, collection_id, chunks_id):
        collection = self.vectorstore.get_collection(collection_id,self.embedding_function)
        deleted_chunks = collection.delete(chunks_id)
        return {"status": "success", "message": "Chunks deleted successfully"}

    def delete_collection(self,collection_id):
        self.vectorstore.delete_collection(collection_id)
        return self.list_collections()
    
    def list_chunks(self,collection_id):
        collection = self.vectorstore.get_collection(collection_id,self.embedding_function)
        return collection.peek()
    
    def mapper_chroma2langchain(self, docs_and_scores):
        """transform chromadb response into langchain documents"""
        langchain_docs = []
        for query_texts, query_metadatas, query_distances in zip(
            docs_and_scores["documents"],
            docs_and_scores["metadatas"],
            docs_and_scores["distances"],
        ):
            query_docs = []
            for document_text, document_metadata, document_distance in zip(
                query_texts, query_metadatas, query_distances
            ):
                doc = Document(
                    page_content=document_text,
                    metadata=copy.deepcopy(document_metadata),
                )
                doc.metadata["distance"] = document_distance
                query_docs.append(doc)
            langchain_docs.append(query_docs)
        return langchain_docs

    def _mapper_unstructured2langchain(self, elements, mode="single"):
        if mode == "elements":
            docs = []
            for element in elements:
                metadata = element.metadata.to_dict()
                # del(meta["file_directory"]) # can be use to hide sensitive info
                metadata["category"] = element.category
                docs.append(Document(page_content=str(element), metadata=metadata))
        elif mode == "paged":
            text_dict = {}
            meta_dict = {}

            for element in elements:
                metadata = element.metadata.to_dict()
                page_number = metadata.get("page_number", 1)

                # Check if this page_number already exists in docs_dict
                if page_number not in text_dict:
                    # If not, create new entry with initial text and metadata
                    text_dict[page_number] = str(element) + "\n\n"
                    meta_dict[page_number] = metadata
                else:
                    # If exists, append to text and update the metadata
                    text_dict[page_number] += str(element) + "\n\n"
                    meta_dict[page_number].update(metadata)

            # Convert the dict to a list of Document objects
            docs = [
                Document(page_content=text_dict[key], metadata=meta_dict[key])
                for key in text_dict.keys()
            ]
        elif mode == "single":
            metadata = elements[0].metadata.to_dict()
            # del(meta["file_directory"]) # can be use to hide sensitive info
            text = "\n\n".join([str(el) for el in elements])
            docs = [Document(page_content=text, metadata=metadata)]
        else:
            raise ValueError(f"mode of {mode} not supported.")
        return docs

    def generate_uuids(self, text_list):
        return [str(uuid.uuid4()) for _ in text_list]
    



class KnowledegeRetrieverClient:
    def __init__(self, url, **kwargs):
        self.base_url = url
        self.headers = {"Content-Type": "application/json"}

    def retrieve_documents(self, user_texts,collection_name, n_results=1):
        url = f"{self.base_url}/query"
        data = {"Texts":user_texts,"N_results": n_results, "Collection_name":collection_name}
        logging.info(f'This is the data object in the Knowledge Retriever client {data}')
        response = requests.post(url, headers=self.headers, data=json.dumps(data))
        answers = []
        if response.status_code == 200:
            response = response.json()
            logging.info(f'This is the answer from the Knowledge Retriever client {response}')

            for answer in response:
                documents = [Document.parse_obj(doc) for doc in answer]
                answers.append(documents)
            return answers
        else:
            raise Exception(f"Request failed with status {response.status_code}")

    def list_collections(self):
        url = f"{self.base_url}/list_collections"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            response = response.json()
        
            return response
        else:
                raise Exception(f"Request failed with status {response.status_code}")


class KnowledgeRetriever:
    def __init__(self, **kwargs):
        self.mode = kwargs["mode"]
        self._default_collection_name = "default_collection"

        if self.mode == "remote":
            self.retriever = KnowledegeRetrieverClient(**kwargs)
        elif self.mode == "local":
            self.retriever = KnowledgeRetrieverLocal(**kwargs)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def retrieve_documents(self, user_texts, collection_name=None, n_results=1, **kwargs):
        if collection_name is None:
            collection_name = self._default_collection_name
        
        docs_and_scores = self.retriever.retrieve_documents(
            user_texts=user_texts,
            collection_name=collection_name,
            n_results=n_results,
            **kwargs,
        )
        return docs_and_scores

    def add_file_to_store(
        self,
        file: bytes,
        collection_name,
        chunk_method_name,
        **kwargs,
    ):
        

        chunk_texts = self.retriever.add_file_to_store(
            file=file, collection_id=collection_name,chunk_method_name=chunk_method_name, **kwargs
        )
        return chunk_texts

    def list_collections(self):
        return self.retriever.list_collections()
    
    def update_chunks(self, collection_name, chunks_ids, chunks):
        return self.retriever.update_chunks(collection_id=collection_name,chunk_ids=chunks_ids, chunks=chunks)
    
    def get_chunks(self,collection_name, chunks_ids):
        return self.retriever.get_chunks(collection_id=collection_name, chunk_ids=chunks_ids)
    
    def list_chunks(self,collection_name):
        return self.retriever.list_chunks(collection_id=collection_name)
    
    def delete_chunks(self,collection_name, chunks_ids):
        return self.retriever.delete_chunks(collection_id=collection_name, chunks_id=chunks_ids)
    
    def delete_collection(self,collection_name):
        return self.retriever.delete_collection(collection_id=collection_name)
    