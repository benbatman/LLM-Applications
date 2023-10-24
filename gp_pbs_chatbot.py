from tools import create_and_store_embeddings, ask_and_get_answer, pretty_print_docs
from chat_streaming import StreamHandler


#################################################################

# TODO: 
# Figure out why second query malfunctions sometimes and doesn't return correct information
# Prompt user to clarify if Ai can't find enough relevant information --- Done
# if user question is too broad prompt to be more specific 
# Figure out how not to stream multiqueries if using the multiquery library
# Should a confidence level be added to the answers?

def main():
    import os 
    #from dotenv import load_dotenv, find_dotenv
    import streamlit as st
    #load_dotenv(find_dotenv(), override=True) # loading openai api key and cohere api key as environment variables
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
    os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY']
    os.environ['PINECONE_ENV'] = st.secrets['PINECONE_ENV']
    os.environ['COHERE_API_KEY'] = st.secrets['COHERE_API_KEY']


    st.set_page_config(
        page_title='Your Custom PBS Assistant',
        page_icon='favicon.ico'
    )

    st.image('PBS_logo.png')
    st.subheader('Your PBS Chatbot powered by GenAI')

    with st.sidebar:
        temperature = float(st.number_input("Creativity level: 0 -> Not creative | 2 -> Very creative", value=0.01, min_value=0.01, max_value=2.0, step=0.1))

        filter_value = st.selectbox("Want to filter the data?", ('title', 'language'), index=None)

    # Create the chunks and vectorestore
    if 'vs' not in st.session_state:
        with st.sidebar:
            with st.spinner('Your PBS chatbot is getting ready'):
                vectorstore = create_and_store_embeddings()
                st.session_state.vs = vectorstore
                
                # tokens, embedding_cost = print_embedding_cost(chunks)
                # st.write(f"Embedding cost: ${embedding_cost:.4f}")
                # st.write(f"Total tokens: {tokens}")

    # Save messages to the session state if not already
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if len(st.session_state.messages) > 4:
        st.session_state.messages = []
        st.session_state.history = ''
    
    chat_history = st.session_state.messages

    question = st.text_input('Chat with your custom PBS assistant!')
    
    if question:
        if 'vs' in st.session_state:
            vectorstore = st.session_state.vs
            chat_box=st.empty()
            stream_handler = StreamHandler(chat_box)
            response, rel_docs, urls, no_context_answer = ask_and_get_answer(vectorstore, question, chat_history=chat_history, temperature=temperature, stream_handler=stream_handler)
            # st.text_area('LLM Answer:', value=response['answer'], height=400)
            #st.write(response['answer'])
            # st.markdown(response['answer'])

            st.subheader("Chat history", divider='blue')
            if 'history' not in st.session_state:
                st.session_state.history = ''

            value = f'Question: {question} \nAnswer: {response["answer"]}'
            st.session_state.history = f'{value} \n {"*"*100} \n {st.session_state.history}'
            h = st.session_state.history
            # st.text_area(label='Chat History', value=h, key='hsistory', height=400)
            st.write(h)

            
            
            st.subheader("Sources", divider='blue')
            st.write(urls)

            # Relevant documents retrieved by similarity search, comment out to remove from printing onto streamlit 
            st.subheader("Documents sourced by AI", divider="blue")
            rel_docs = pretty_print_docs(rel_docs)
            st.write(rel_docs)
            #st.text_area("Docs", value=rel_docs)

            chat_history.append((question, response['answer']))
            st.session_state.messages = chat_history

###############################################

if __name__ == "__main__":
   main()
