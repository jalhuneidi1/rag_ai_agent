from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate # import chat prompt template

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about books. 

Here are some books: {books}

Here is the question to answer: {question}
"""

# Chat prompt template that takes in the books and question variables for the model to respond to them
prompt = ChatPromptTemplate.from_template(template)

# invoke chain that combines multiple things together to run LLM
# prompt taken from template and passed to the model
chain = prompt | model

result = chain.invoke({"books": [], "question": "What are the famous opening words of A Tale of Two Cities that set up the paradoxical nature of the era?"})
print(result)