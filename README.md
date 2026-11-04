# Production Ready Retrieval Augmented Generation (RAG) AI Agent

Large Language Models (LLMs) are trained on a large but fixed corpus of data, limiting their ability to reason about private or recent information. Fine-tuning is one way to mitigate this, but is often not well-suited for factual recall and can be costly. Retrieval augmented generation (RAG) has emerged as a popular and powerful mechanism to expand an LLM's knowledge base, using documents retrieved from an external data source to ground the LLM generation via in-context learn

This project is a local Retrieval-Augmented Generation (RAG) system. It scans a local repository of Markdown files, parses and fragments them structurally to preserve layout context, converts them into numeric semantic embedding, and indexes them into an isolated vector database. Users can run conversational inquiries against their library via an interactive command-line interface running locally optimized language models.

## System Architecture
```text
+-------------------------------------------------------+
|                  1. Data Layer                        |
|             (Raw Markdown Documents)                  |
+--------------------------+----------------------------+
                            |
                            | [Directory & Text Loaders]
                            |
                            v 
+-------------------------------------------------------+
|                 2. Processing Layer                   |
|     (Markdown Structural Splitting & Tokenization)    |
+--------------------------+----------------------------+
                            |
                            | [Ollama API / Vector Inference]
                            |
                            v 
+-------------------------------------------------------+
|            3. Storage & Retrieval Layer               |
|           (Local Chroma DB Vector Store)              |
+---------------------------+---------------------------+
                            |
                            | [Context Injection via Vector Query]
                            |
                            v 
+-------------------------------------------------------+
|           4. Execution & Orchestration Layer          |
|       (LangChain Expressions & Llama 3.2 UI Loop)     |
+-------------------------------------------------------+
```

## Core Dependencies and Components
1. Runtime Environment & Client Models
- Ollama Ecosystem: Manages low-latency local model orchestration and weights optimization.
	- Inference Model (llama3.2): A lightweight, 3-billion parameter model highly optimized for local compute, text summarization, and context-dependent QA tasks.
- Embedding Model (mxbai-embed-large): A specialized text embedding model with a 512-token context length. It maps raw text snippets into 1024-dimensional semantic vectors.
- langchain_ollama SDK: The glue framework that wraps Ollama's local endpoints into unified object APIs (OllamaLLM and OllamaEmbeddings) for pipeline execution.

2. Vector Indexing Engine
- ChromaDB (langchain_chroma): An embedded vector database designed to reside inside an application process. It handles local SQLite storage arrays, coordinates high-dimensional metadata indices, and exposes fast cosine similarity queries via distance calculation abstractions.

3. Extraction & Tokenization Pipelines
- DirectoryLoader & TextLoader (langchain_community): Scans file structures recursively using specific glob filtering patterns (**/*.md). It unifies filesystem content streams directly into structured objects.
- MarkdownTextSplitter (langchain_text_splitters): A structural text tokenizer. Unlike standard word-count wrappers, it maps chunk separations around native Markdown structural landmarks (e.g., # headers, list symbols, and linebreaks) to avoid dropping key paragraph attributes mid-sentence.

## Implementation Blueprint
```text
rag_ai_agent
├── chroma/
│   └── chroma.sqlite3
├── data/
│   └── books/
│   	└── a_tale_of_two_cities.md
│   	└── alice_in_wonderland.md
│   	└── beautiful_stories_from_shakespeare.md
│   	└── the_great_gatsby.md
│   	└── the_pictures_of_dorian_grey.md
│   	└── the_wonderful_wizard_of_oz.md
│   	└── winnie_the_pooh.md
│   	└── wuthering_heights.md
├── main.py
├── vector.py
└── requirements.txt
```
