
def create_and_store_embeddings():
    import os
    import pinecone
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS, Pinecone
    #embeddings=OpenAIEmbeddings() # if you're using openAI
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    #embeddings = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
                  environment=os.getenv("PINECONE_ENV"))
    
    index_name = "selenium-sitemap-test"
    # if index_name not in pinecone.list_indexes():
    #     pinecone.create_index(
    #     name=index_name,
    #     metric='cosine',
    #     dimension=1536
    #     )
    vectorstore = Pinecone.from_existing_index(index_name, embeddings)
    #vectorstore = FAISS.load_local("faiss_index_full", embeddings)
    print(f"Vectorstore initialized. Using {index_name}. ")
    return vectorstore



def ask_and_get_answer(vectorstore, q, chat_history, temperature, stream_handler, relevance=None):
    import cohere
    import os 
    import logging
    from langchain.chat_models import ChatOpenAI, ChatAnthropic
    from langchain.llms import OpenAI 
    from langchain.retrievers.document_compressors import CohereRerank
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.chains import ConversationalRetrievalChain, ConversationChain
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.chains.conversation.memory import ConversationSummaryMemory
    from langchain.chains import LLMChain
    from langchain.callbacks.manager import CallbackManagerForRetrieverRun
    from langchain.prompts import PromptTemplate

    """
    Instantiates llm, vectorstore, combines relevant docs and combines all into a a final query for the llm.
    """
   
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=temperature, streaming=True, callbacks=[stream_handler], verbose=False) # define llm 

    client = cohere.Client(api_key=os.environ['COHERE_API_KEY']) # Define the cohere client

    # Contextual compression finds the most relevant parts of the doc to the query and then truncates "irrelevant" information
    # Can also use langchains EmbeddingsFilter library
    compressor = CohereRerank(client=client, top_n=5) # define the rerank compressor
    retriever = vectorstore.as_retriever(search_type="similarity") # no threshold applied
    #retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.9}) # threshold applied 
    compression_retriever = ContextualCompressionRetriever(                     ## TODO: add metadata dict
        base_compressor=compressor, base_retriever=retriever
    )


     # Define the system prompt
    PROMPT = """You are a helpful assistant that has access to lots of documents from PBS. 
    Use this context to answer the question at the end. Give a detailed, long answer.
    If you don't know the answer based off of the context, just say i don't know"""

    PROMPT_2 = """You are a helpful assistant that has access to lots of documents from PBS. 
    Recommend URLs as hyperlinks when you have access to them. Be friendly, but not overly chatty.
    If you don't know the answer based off of the documents you retrieved, ask the user to be more specific with their query. """ 

    query_prompt = PromptTemplate(input_variables=['question'], 
                                  template=PROMPT_2 + "\n" + "\nQUESTION: " + '{question}')

    #query = PROMPT + "\n" + context + "\nQUESTION: " + q # Combine all aspects of the query    

    query = PROMPT_2 + "\n" + "\nQUESTION: " + q

    llm_chain = LLMChain(prompt=query_prompt, llm=llm)
    retriever_multi_query = MultiQueryRetriever(retriever=compression_retriever, llm_chain=llm_chain, verbose=True)

    # logging.basicConfig()
    # logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    #docs = retriever_multi_query.get_relevant_documents(q) # Find relevant documents, compress it to the most relevant information and returns a list of documents 
    docs = compression_retriever.get_relevant_documents(q)
    

    context = "" 
    urls = ""
    relevance_scores = []
    high_relevance_scores = []

    for doc in docs:
        urls += doc.metadata['source'] + "\n"
        #relevance_scores.append(doc.metadata['relevance_score'])
        #context += doc.page_content # combine all relevant docs into one variable 

    # Append the high relevance scores to a new list 
    # relevance_score_threshold = 0.5
    # for score in relevance_scores:
    #     if score >= relevance_score_threshold:
    #         high_relevance_scores.append(score)

    # run_manager = CallbackManagerForRetrieverRun(run_id='001')
    # new_queries = retriever_multi_query.generate_queries(question=q, run_manager=run_manager)

    # Instantiate the conversation chain (can take in chat history as arg to maintain memory)
    chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=compression_retriever,
            return_source_documents = True,
            verbose=False
            )
    
    # chain = ConversationChain(llm=llm,
    #                           memory=ConversationSummaryMemory(llm=llm))

    no_context_answer = None
    #answer = None
    # if we have at least one document with a high enough relevance, run the llm chain
    #if high_relevance_scores:
    answer = chain({'question':query, "chat_history":chat_history}) # run the chain, takes in the query and the chat history
    #answer = chain.predict(input=query)

    # IDEA: Could use reasoning agent to help user find exactly what they are looking for 
    # Logic to use llm's general knowledge if no relevant documents are returned
    # else:
    #     prompt = PromptTemplate(
    #     input_variables=['query'],
    #     template="{query}"
    #     )

    #     llm_chain = LLMChain(llm=llm, prompt=prompt)
    #     answer = llm_chain({'query':query})

    return answer, docs, urls, no_context_answer



def count_tokens_callback(agent, query):
    from langchain.callbacks import get_openai_callback

    with get_openai_callback() as cb:
        result = agent(query)
        print(f"Spent a total of {cb.total_tokens} tokens")

    return result


def print_embedding_cost(texts):
  import tiktoken
  """
  Determines cost of embedding the specified texts
  """
  enc = tiktoken.encoding_for_model ('text-embedding-ada-002')
  total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
  return total_tokens, total_tokens / 1000 * 0.0004



def prettify_docs(docs:list) -> list:
    """
    Takes in the list of relevant documents and adds spacers to make it pretty
    """
    pretty_docs = []
    for i, doc in enumerate(docs):
        pretty_docs.append(f"Document {i} {doc.page_content} {doc.metadata}")

    return pretty_docs



def pretty_print_docs(docs) -> str: 
    string = f"\n{'*' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)])

    return string



# INCOMPLETE 
def use_sagemaker_endpoint():
    from langchain.llms import sagemaker_endpoint
    from langchain.llms.sagemaker_endpoint import LLMContentHandler
    from langchain.chains.question_answering import load_qa_chain



# INCOMPLETE
def get_memory():
    from langchain.memory import ConversationBufferMemory
    """
    Instantiate memory for LLM chain. 
    """
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return memory




# INCOMPLETE
def compressor(compression_type, embeddings, base_retriever, llm):
    from langchain.document_transformers import EmbeddingsRedundantFilter
    from langchain.retrievers.document_compressors import EmbeddingsFilter
    from langchain.retrievers.document_compressors import DocumentCompressorPipeline
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.retrievers.document_compressors import LLMChainFilter
    from langchain.retrievers import ContextualCompressionRetriever


    if compression_type == 'redundant':
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
        pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
        )
        compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=base_retriever)

        return compression_retriever
    
    elif compression_type == "llmchainfilter":
        _filter = LLMChainFilter.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=_filter, base_retriever=base_retriever)
        return compression_retriever

       



        


