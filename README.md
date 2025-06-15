## Simple RAG App with LangChain & Streamlit

![Screenshot 2025-06-15 at 4 19 58 PM](https://github.com/user-attachments/assets/03754896-7989-4cc7-ada8-948a3405483f)


Build your own Retrieval-Augmented Generation (RAG) app using LangChain, OpenAI, and ChromaDB — all wrapped in a clean and interactive Streamlit UI.

This project walks you through:

    Loading and chunking a PDF textbook 

    Creating a local vector store using ChromaDB

    Embedding text with OpenAI Embeddings

    Performing vector similarity search 🔍

    Creating a chat app that understands contextual follow-up questions using LangChain's RAG chain and memory 🎯

## Use Case:

Ask questions about your own proprietary data, like internal documentation or textbooks, and get LLM-powered responses backed by real context.

## Get Started:

```graphql
.
└── andy
    ├── indexing.py              # Loads and chunks the PDF, builds vector DB
    ├── keys.yml                 # Stores your OpenAI API key
    ├── querying.py              # Streamlit app that queries the vector DB
    ├── requirements.txt         # Python dependencies
    └── statistics_txtbook.pdf   # Data source (your textbook)


```

## Instructions


1. Clone the repository
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name/andy
```
2. Install dependencies
Use your preferred virtual environment tool, then:
```
pip install -r requirements.txt
```
3. Add your OpenAI API Key
Create a file named keys.yml in the same directory as indexing.py, and add:
```
openai: <your-openai-api-key>
```

4. Create the Vector Database
This step reads the textbook, chunks it, embeds it, and stores it in a local vector DB.
You should see printouts showing the chunking process and document batches being added.
```
python indexing.py
```


5. Run the RAG App, once the vector DB is created, launch the chat UI:
Use the sidebar to pick your model, and ask anything from the textbook — follow-up questions are supported too!
```
streamlit run querying.py
```


