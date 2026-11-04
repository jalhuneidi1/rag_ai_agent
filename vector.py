# Logic for embedding documents and retrieving relevant information based on a query to be passed to the model using LangChain and OllamaLLM.
# Database hosted on Ollama's vector database ChromaDB (local host), which is a vector database that allows for efficient storage and retrieval of high-dimensional vectors.

import os 
from langchain_ollama.llms import OllamaEmbeddings # takes text and converts it into a vector representation (numbers) to look up data efficiently
from langchain_chroma import Chroma # vector database that allows for efficient storage and retrieval of high-dimensional vectors
from langchain_community.document_loaders import DirectoryLoader # loads documents from a directory and converts them into a format that can be used by the vector database
from langchain_community.document_loaders import TextLoader # loads text files and converts them into a format that can be used by the vector database
from langchain_text_splitters import MarkdownTextSplitter # splits text into smaller chunks to be used by the vector database

# define/create an instance of the OllamaEmbeddings class to convert text into vector representation to match the setup of the vector database, which is used to look up data efficiently. The model used is "mxbai-embed-large" which is a large embedding model that can handle large amounts of text data and convert it into a vector representation for efficient storage and retrieval in the vector database.
embedding = OllamaEmbeddings(model = "mxbai-embed-large") 

db_location = "./chroma_langchain_db"
#checks if the database location exists, if not it will create a new database and add the documents to it, if it does exist it will just use the existing database and not add the documents again
add_db = not os.path.exists(db_location)


# Initialize Chroma vector database to to store the book data
vector_store_db = Chroma(
    collection_name="books", persist_directory=db_location, embedding_function=embedding
                         ) 

 # create vector store and automaticallyadd the documents to the vector database, only load and chunk makedown files if the databases does not exist, if it does exist it will just use the existing database and not add the documents again
if not add_db:
    print("Initializing vector database and adding documents... Processing files found in 'books' directory...")
    # Load documents from the "books" directory and convert them into a format that can be used by the vector database
    loader = DirectoryLoader("data/books",
                             glob="**/*.md",
                             loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"}
                             )
    documents = loader.load() # load the documents from the "books" directory and convert them into a format that can be used by the vector database

# check if any documents were found in the "books" directory, if not it will print a message to the user to add markdown files to the directory and try again
    if documents:
        # split the documents into smaller chunks to be used by the vector database, with a chunk size of 1000 characters and an overlap of 100 characters between chunks to ensure that the context is preserved when splitting the documents into smaller chunks
        text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # iterate through the chunksand add them to the vector database, with the metadata of each chunk being the source file name and the content of each chunk being the text of the chunk
        # LanhChain's loader populates document metadata with the source file nameto be used to identify the source of each chunk when adding them to the vector database
        for index, chunk in enumerate(chunks):
           file_path = chunk.metadata.get("source", "unknown_file")
           file_name = os.path.basename(file_path)
           book_title, _ = os.path.splitext(file_name)

           # Enrich metadata payload for prompt formatting
           chunk.metadata["title"] = book_title.replace("_", " ").title()  # Format book title for better readability
           chunk.metadata["author"] = "Unknown"  # Placeholder for author, can be updated if author information is available

        # Add the chunks to the vector database
        print(f"Adding {len(chunks)} chunks to the vector database...")
        vector_store_db.add_documents(chunks)
        print("Documents added to the vector database successfully.")
    else:
        print("No documents found in the 'books' directory. Please add markdown files to the directory and try again.")
else:
    print("Existing vector database found.  Loading stored vectors directly.")


# make datastore usable by LLM by creating a retriever that can be used to retrieve relevant information (look up documents) from the vector database based on a query, with a maximum of 5 results returned
retriver = vector_store_db.as_retriever(search_kwargs={"k": 5})