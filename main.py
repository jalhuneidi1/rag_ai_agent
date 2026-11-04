from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate # import chat prompt template
from vector import retriver # import retriever from vector.py to retrieve relevant information from the vector database

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about books. 

Here are some books: {books}

Here is the question to answer: {question}
"""

# Chat promtp template that takes in the books and question variables for the model to respond to them
prompt = ChatPromptTemplate.from_template(template)

# invoke chain that combines multiple things together to run LLM
# prompt taken from template and passed to the model
chain = prompt | model

while True:
    print("\n\n----------------------------------------------------")
    question = input("Enter your question (or type 'exit' to quit): ")
    print("\n\n")
    if question.lower() == 'exit':
        break
    
    # use retriever to retrieve relevant information from the vector database based on the question, with a maximum of 5 results returned then use as a parameter to the prompt template and model to get the answer
    book_content = retriver.invoke(question)
    result = chain.invoke({"books": book_content, "question": question})
    print(result)