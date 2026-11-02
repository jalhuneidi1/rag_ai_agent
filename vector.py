# Logic for embedding documents and retrieving relevant information based on a query to be passed to the model using LangChain and OllamaLLM.
# Database hosted on Ollama's vector database ChromaDB (local host), which is a vector database that allows for efficient storage and retrieval of high-dimensional vectors.

from langchain_ollama.llms import OllamaEmbeddings # takes text and converts it into a vector representation (numbers) to look up data efficiently
from langchain_chroma import Chroma # vector database that allows for efficient storage and retrieval of high-dimensional vectors
from langchain_core.documents import Document # represents a document in the vector database
import os
# import pandas as pd

df = 