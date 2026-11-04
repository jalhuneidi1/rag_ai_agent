from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate # import chat prompt template
from vector import retriver # import retriever from vector.py to retrieve relevant information from the vector database

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about books. 

Review the books in the digital library: {books}

Using only the digital library, answer the following question(s): {question}
"""

# Chat prompt template that takes in the books and question variables for the model to respond to them
prompt = ChatPromptTemplate.from_template(template)

# invoke chain that combines multiple things together to run LLM
# prompt taken from template and passed to the model
chain = prompt | model

# the following function extracts the text and title metadata fromt he LangChain Document objects and structures them cleanly for the prompt context.
def format_retireved_books(docs):
    formatted_books = []
    # iterate through the retrieved documents and extract the title and content metadata, then format them into a string for the prompt context
    for doc in docs:
        title = doc.metadata.get("title", "Unknown Book")
        content = doc.page_content.strip()

        formatted_books.append(f"Title: {title}\nContent: {content}")
    return "\n\n".join(formatted_books)

while True:
    print("\n\n----------------------------------------------------")
    question = input("Enter your question (or type 'exit' to quit): ")
    print("\n\n")
    if question.lower() == 'exit':
        break

    # fetch the relevant book content from the vector database using the retriever and format it for the prompt context as a clean unified markdown string then pass the formatted string to the model along with the question to get the answer from the model
    raw_book_content = retriver.invoke(question)
    book_content = format_retireved_books(raw_book_content)
    result = chain.invoke({"books": book_content, "question": question})
    print(result)