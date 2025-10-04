from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatOpenAI()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# RAG

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import os

# Global variable to store vector store
vector_store = None

def process_pdf(pdf_path):
    global vector_store
    
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    
    return len(splits)

def get_relevant_context(query):
    global vector_store
    if vector_store is None:
        return ""
    
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context


def chat_node(state: ChatState):
    messages = state['messages']
    
    # Get the last user message
    last_message = messages[-1].content if messages else ""
    
    # Get relevant context from PDF if available
    context = get_relevant_context(last_message)
    
    # If context exists, add it to the message
    if context:
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the following context to answer the user's question. If the answer is not in the context, say so.\n\nContext:\n{context}"),
            ("human", "{question}")
        ])
        
        enhanced_message = rag_prompt.format(context=context, question=last_message)
        messages[-1].content = enhanced_message
    
    response = llm.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

