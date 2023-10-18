import streamlit as st 
import json
import pandas as pd
from streamlit_chat import message # if you want true chatbot style UI 
from sentence_transformers import SentenceTransformer
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.vectorstores.pgvector import PGVector 
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)

#################################################################

# TODO: put all helper functions into tools.py file 

def create_and_store_embeddings():
    #embeddings=OpenAIEmbeddings() # if you're using openAI
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    #embeddings = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings)
    return vectorstore

def get_memory():
    """
    Instantiate memory for LLM chain. 
    """
    memory = ConversationBufferMemory(memory_key='history', return_messages=True)

def ask_and_get_answer(vectorstore, q, chat_history, k=5):
   
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)

    docs = vectorstore.similarity_search(q) # Find relevant documents 

    context = "" 

    doc_sources_string = ""
    for doc in docs:
        #doc_sources_string += docs.metadata['sources'] + "\n"
        context += doc.page_content # combine all relevant docs into one variable 

    # Instantiate the conversation chain (can take in chat history as arg to maintain memory)
    chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_type="similarity"),
            #memory=get_memory(),
            return_source_documents = True
            )
    
    # Define the system prompt
    PROMPT = """You are a helpful assistant that has access to lots of documents from PBS. 
    You this context to answer the question at the end. Give a detailed, long answer.
    Recommend URLs where applicable"""

    query = PROMPT + "\n" + f"{context}" + "\n" + q # Combine all aspects of the query

    answer = chain({'question':query, "chat_history":chat_history}) # run the chain, takes in the query and the chat history

    return answer, docs

def print_embedding_cost(texts):
  import tiktoken
  enc = tiktoken.encoding_for_model ('text-embedding-ada-002')
  total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
  return total_tokens, total_tokens / 1000 * 0.0004



def main():
    import os 
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)


    st.set_page_config(
        page_title='Your Custom PBS Assistant',
        page_icon='favicon.ico'
    )

    st.image('PBS_logo.png')
    st.subheader('Your PBS Chatbot powered by GenAI')

     
    # Create the chunks and vectorestore
    with st.spinner('Your PBS chatbot is getting ready'):
        vectorstore = create_and_store_embeddings()
        st.session_state.vs = vectorstore
        
        # tokens, embedding_cost = print_embedding_cost(chunks)
        # st.write(f"Embedding cost: ${embedding_cost:.4f}")
        # st.write(f"Total tokens: {tokens}")

    # Save messages to the session state 
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'memory' not in st.session_state:
        st.session_state.memory = get_memory()

    chat_history = st.session_state.messages

    q = st.text_input('Ask a question')

    with st.spinner('Working on your request'):
        if q:
            if 'vs' in st.session_state:
                vectorstore = st.session_state.vs
                response, docs = ask_and_get_answer(vectorstore, q, chat_history=chat_history)
                st.text_area('LLM Answer:', value=response['answer'])
                st.divider()
                st.text_area('Source documents', value=docs)
                chat_history.append((q, response['answer']))
                st.session_state.messages = chat_history


                st.divider()
                if 'history' not in st.session_state:
                    st.session_state.history = ''

                value = f'q: {q} \nA: {response["answer"]}'
                st.session_state.history = f'{value} \n {"*"*100} \n {st.session_state.history}'
                h = st.session_state.history
                st.text_area(label='Chat History', value=h, key='history', height=400)


    # for i, msg in enumerate(st.session_state.messages[1:]):
    #     if i % 2 == 0:
    #         message(msg.content, is_user=True)
    #     else:
    #         message(msg.content, is_user=False)




###############################################


if __name__ == "__main__":
   main()