"""streamlit run AL_streamlit.py to run the streamlit application,
this example runs the RAG application with history summarization, 
it's a complete working example
"""

from langchain_chroma import Chroma 
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


import streamlit as st
import yaml
import os



RAG_DB = "./statisticsKnowledge.db"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODELS = ["gpt-4o-mini","gpt-4o"] 
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open("./keys.yml"))['openai']


#browser header name + body structure
st.set_page_config(page_title="My RAG Application",layout="wide")
st.title("CUSTOM RAG APPLICATION")

# Sidebar for model selection
model_selection = st.sidebar.selectbox(
    label = "Choose your LLM",
    options = LLM_MODELS,
    index=0
)


#set up history 
msgs = StreamlitChatMessageHistory(key = "langchain_msgs")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_msgs = st.expander("view the message contents in session state")

def init_rag_chain():

    embedding_client = OpenAIEmbeddings(
        model = EMBEDDING_MODEL
    )

    vector_db = Chroma(
       embedding_function=embedding_client,
       persist_directory=RAG_DB,
       collection_name = "stats"
    )

    # Print out how many documents are stored in the collection
    doc_count = vector_db._collection.count()
    st.write(f"Number of documents in vectorstore: {doc_count}")



    retriever = vector_db.as_retriever()

    llm = ChatOpenAI(
        model=model_selection,
        temperature=0.7
    )

    #rewrites user question so that LLM understands, uses chat history not rag db
    contextualize_q_system_prompt = """
    Given the chat history and the latest user input 
    rewrite it as a standalone question that captures the full context. 
    Do not answer the question, just rewrite it. if it's already standalone, 
    just return it as is.
    """ 

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    ###This history_aware_retriever is responsible for the following
    #1. takes the latest user question and chat history to rewrite a standalone question if it needs
    #2. sends the standalone question it wrote to Chroma to fetch for relevant documents
    history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
    
    
    ##################################
    qa_system_prompt = """
    You are an assitant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    {context}
    """ 
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    ###question_answer_chain is
    #1. just a template that tells the LLM what to do 
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


    # Combines the history_aware_retriever + answering logic 
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )



custom_rag_model = init_rag_chain()

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content) 

question = st.chat_input("Enter your question here:", key = "query_input")

if question: 
    with st.spinner("Thinking ..."):
        st.chat_message("human").write(question)

        response = custom_rag_model.invoke(
            {"input":question},
            config ={
                "configurable":{"session_id":"any"}
            }
        )

        st.chat_message("ai").write(response["answer"])

with view_msgs:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_msgs.json(st.session_state.langchain_msgs)