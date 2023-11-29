"""Load html from files, clean up, split, ingest into Weaviate."""
# import pickle
from pathlib import Path

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm

from config import settings
from vectorstore.documentretriever import KnowledgeRetriever
from embedding.embedders import Embedding

#TODO use fsspec for Minio interface


hf_embeddings_model = settings["embeddings"]["embedding_model"]
chunk_size = settings["vectorstore"]["chunks_size"]
chunk_overlap = settings["vectorstore"]["chunks_overlap"]
data_directory = settings["vectorstore"]["data_path"]
data_directory = Path(data_directory)
print("data dir", data_directory.absolute())
ext = "pdf"

# TODO

def get_ext_paths(dir_path, ext):
    """Return a list of paths for all file that is of type ext"""
    if ext[0] == ".":
        ext = ext[1:]
    ext_pattern = f"*.{ext}"
    pdfs = []
    for path in Path(dir_path).rglob(ext_pattern):
        pdfs.append(str(path.absolute()))
    return pdfs

def ingest_docs():
    """Get documents from web pages."""
    paths = get_ext_paths(data_directory, ext)
    print("paths",paths)
    documents_used = 0

    embedder = Embedding(**settings["embeddings"])
        
    retriever = KnowledgeRetriever(embedding_function=embedder.vectorize)
    for path in tqdm(paths):
        loader = PyMuPDFLoader(path)
        raw_documents = loader.load()

        for i, document in enumerate(raw_documents):
            metadata = document.metadata
            source_path = Path(metadata["source"])
            relative_path = source_path.relative_to(data_directory.absolute())
            metadata["source"] = str(relative_path)
            metadata["file_path"] = str(relative_path)
            raw_documents[i].metadata = metadata
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        documents = text_splitter.split_documents(raw_documents)
        if not documents:
            continue
        documents_used += 1
        retriever.add(documents)
    print("documents_used", documents_used)
    # Save vectorstore
    retriever.vectorstore.persist()


if __name__ == "__main__":
    ingest_docs()
