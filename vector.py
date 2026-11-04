# Logic for embedding documents and retrieving relevant information based on a query to be passed to the model using LangChain and OllamaLLM.
# Database hosted on Ollama's vector database ChromaDB (local host), which is a vector database that allows for efficient storage and retrieval of high-dimensional vectors.

from langchain_ollama.llms import OllamaEmbeddings # takes text and converts it into a vector representation (numbers) to look up data efficiently
from langchain_chroma import Chroma # vector database that allows for efficient storage and retrieval of high-dimensional vectors
from langchain_core.documents import Document # represents a document in the vector database
import os
import pandas as pd

df = pd.read_csv("books.csv") # read in the books.csv file that contains the book data
embedding = OllamaEmbeddings(model = "mxbai-embed-large") # define/create an instance of the OllamaEmbeddings class to convert text into vector representation

db_location = "./chroma_langchain_db"
add = not os.path.exists(db_location)

# Create a Chroma vector database to store the book data, prepare/convert data in documents format, and add it to the database if it doesn't already exist
if add_document: 
    documents = []
    ids = []
    # Loop through each row in the DataFrame and create a individual Document object for each book, then add it to the documents list
    for index, row in df.iterrows():
        document = Document(page_content=row['description'], metadata={"title": row['title'], "author": row['author']},
        id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store_db = Chroma(collection_name="books", persist_directory=db_location, embedding_function=embedding) # create a Chroma vector database to store the book data

if add_document:
    vector_store_db.add_documents(documents=documents, ids=ids) # create vector store and automaticallyadd the documents to the vector database

# make datastore usable by LLM by creating a retriever that can be used to retrieve relevant information (look up documents) from the vector database based on a query, with a maximum of 5 results returned
retriver = vector_store_db.as_retriever(search_kwargs={"k": 5})