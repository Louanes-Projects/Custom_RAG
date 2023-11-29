import uuid
import spacy
import numpy as np

from langchain.text_splitter import (
    TokenTextSplitter as langchain_TokenTextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from enum import Enum
from pydantic import BaseModel, Field


class ChunkMethod(str, Enum):
    method1 = "token_text_splitter"
    method2 = "nltk_text_splitter"
    method3 = "recursive_character_text_splitter"
    method4 = "spacy_text_splitter"
    method5 = "clustering_adjacent_sentences"


class Document(BaseModel):
    """Langchain Interface for interacting with a chunk."""

    page_content: str
    metadata: dict = Field(default_factory=dict)

class ChunkSplitter:
    def __init__(self,documents):
        self.documents=documents

    def split_with_token_text_splitter(self, chunk_size=500, chunk_overlap=100):
        text_splitter = langchain_TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return self.map2langchaindocuments(text_splitter)

    def split_with_nltk_text_splitter(self,chunk_size=500, chunk_overlap=100, nltk_separator="\n\n"):
        nltk_splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=nltk_separator)
        return self.map2langchaindocuments(nltk_splitter)

    def split_with_spacy_text_splitter(self,chunk_size=500, chunk_overlap=100, spacy_separator="\n\n", spacy_pipeline="fr_core_news_sm"):
        spacy_splitter = SpacyTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=spacy_separator, pipeline=spacy_pipeline)
        return self.map2langchaindocuments(spacy_splitter)

    def split_with_recursive_character_text_splitter(self,chunk_size=500, chunk_overlap=100, length_function=len, separators=["\n\n", "\n", " ", ""]):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=length_function , separators=separators)
        return self.map2langchaindocuments(text_splitter)

    def split_with_Sentence_Transformers_Token_Text_Splitter(self, chunk_size=500, chunk_overlap=50, model_name="sentence-transformers/all-mpnet-base-v2"):
        text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, model_name=model_name)
        return self.map2langchaindocuments(text_splitter)

    # def split_with_unstructured(self , strategy="fast", mode="elements"):
    #     loader = UnstructuredPDFLoader(self.input_directory + self.file_name, strategy=strategy, mode=mode)
    #     docs = loader.load()
    #     doc_chunks = [doc.page_content for doc in docs]
    #     return self._write_chunks_to_file(doc_chunks, "splitting_6_unstructured.txt")

    def split_with_clustering_adjacent_sentences(self, min_chars=60, max_chars=3000, pipeline="fr_core_news_sm"):
        
        def process(text):
            nlp = spacy.load(pipeline)
            doc = nlp(text)
            sents = list(doc.sents)
            vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])
            return sents, vecs

        def cluster_text(sents, vecs, threshold):
            clusters = [[0]]
            for i in range(1, len(sents)):
                if np.dot(vecs[i], vecs[i-1]) < threshold:
                    clusters.append([])
                clusters[-1].append(i)
            return clusters

        def clean_text(text):
            return text

        threshold = 0.3
        sents, vecs = process(self.documents[0].page_content)
        clusters = cluster_text(sents, vecs, threshold)

        documents = []
        for cluster in clusters:
            cluster_txt = clean_text(' '.join([sents[i].text for i in cluster]))
            cluster_len = len(cluster_txt)

            if cluster_len < min_chars:
                continue
            elif cluster_len > max_chars:
                threshold = 0.6
                sents_div, vecs_div = process(cluster_txt)
                reclusters = cluster_text(sents_div, vecs_div, threshold)

                for subcluster in reclusters:
                    div_txt = clean_text(' '.join([sents_div[i].text for i in subcluster]))
                    div_len = len(div_txt)

                    if div_len < min_chars or div_len > max_chars:
                        continue

                    doc = Document(page_content=div_txt, metadata={"length": div_len})
                    documents.append(doc)
            else:
                doc = Document(page_content=cluster_txt, metadata={"length": cluster_len})
                documents.append(doc)

        return documents
    
    def map2langchaindocuments(self, splitter):
        """returns a list of langchain documents aka chunks."""
        print(type(splitter))
        chunks = splitter.split_documents(self.documents)
        return chunks


    def split_text(self, method, **kwargs):
        method_mapping = {
            "token_text_splitter": self.split_with_token_text_splitter,
            "nltk_text_splitter": self.split_with_nltk_text_splitter,
            "spacy_text_splitter": self.split_with_spacy_text_splitter,
            "recursive_character_text_splitter": self.split_with_recursive_character_text_splitter,
            "clustering_adjacent_sentences": self.split_with_clustering_adjacent_sentences,
            "sentence_transformers_token_text_splitter": self.split_with_Sentence_Transformers_Token_Text_Splitter,
        }

        if method not in method_mapping:
            raise ValueError(f"Invalid method: {method}")

        return method_mapping[method](**kwargs)
    


    