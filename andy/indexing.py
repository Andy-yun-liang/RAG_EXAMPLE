

from langchain_community.document_loaders import PyPDFLoader

txtbook_path = "./statistics_txtbook.pdf"
loader = PyPDFLoader(txtbook_path)

docs = loader.load()

print(f"Our original textbook has {len(docs)} pages")

#skipping preface,table of contents, and index 
#each document is a single page with varying length of text
#so we are removing pages (1-11 inclusive both ends) and (603-613 inclusive both ends)

cleaned_text_book = []
exclude_pages = set(range(0,12)) | set(range(603,614))
for doc in docs:
    page_num = doc.metadata.get("page",-1)
    if page_num in exclude_pages:
        continue
    cleaned_text_book.append(doc)
print(f"we are removing table of contents and index, we have {len(cleaned_text_book)} pages")


#remove pages with less than 100 words
useless_pages = []
filtered_docs = []
for doc in cleaned_text_book:
    page_num = doc.metadata.get("page",-1)
    if len(doc.page_content.split()) < 100:
        useless_pages.append((page_num,doc))
    else:
        filtered_docs.append(doc)

print(f"removing pages with less than 100 words, we have {len(filtered_docs)} pages")


#decide if we need to chunk
lengths = [len(doc.page_content) for doc in filtered_docs]
print(f"Average length: {sum(lengths)/len(lengths):.2f} characters")
print(f"Max length: {max(lengths)} characters")


# I will chunk cuz the max length is like 31k....
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_chunker = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 500,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

docs_chunked = text_chunker.split_documents(filtered_docs)

print(f"we now have {len(docs_chunked)} pages because we reduced characters per page")


#choose a text embedding model
text_embedding_model = "text-embedding-3-small"


#import text embedding client
from langchain_openai import OpenAIEmbeddings

#import you key
import os , yaml
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open("keys.yml"))['openai']


#make sure you have your key in your environment before you run this
embedding_function = OpenAIEmbeddings(
    model=text_embedding_model,
)


# OpenAi has a 300k token limit, meaning that you can only upload 300k worth of token of 
# documents at once, so to bypass that I am going to make multiple uploads in smaller sizes

batch_size = 100
initial_batch = docs_chunked[:batch_size]


from langchain_chroma import Chroma
#choose and create a vector database, I chose Chroma just cuz its the most used.
"""
document : gives the initial structure to the db
embedding: converts text into vectors
persist_directory: where we want to place the path of the vector database, also give it a name
(Note: it creates a new vector db if you misname this lol)
collection_name: essentially a table name for the document(s) cuz you can have different subjects,
not sure why you would want to but its there.
"""
vectorstore = Chroma.from_documents(
    documents=initial_batch,
    embedding=embedding_function,   
    persist_directory="statisticsKnowledge.db",  
    collection_name = "stats"
)

def batching_chunks(docs,size):
    for i in range(0,len(docs),size):
        yield docs[i:i+size]

#make sure to skip first batch
for i,chunk_batch in enumerate(batching_chunks(docs_chunked[batch_size:], batch_size)):
    print(f"Adding batch {i + 2} of {len(docs_chunked) // batch_size + 1}")
    vectorstore.add_documents(chunk_batch)

#test vector db with simple similar search
"""
We are checking if the llm can give us the pages that correlate to our question,
I am only going to take top 3 pages that is most similar to my query.
"""
quick_sanity_check = vectorstore.similarity_search(
    query="When working with advertising data, what are potential predictor and response variables?", 
    k = 3
)

import pprint

pprint.pp(quick_sanity_check[0].page_content)