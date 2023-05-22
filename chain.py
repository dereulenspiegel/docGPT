from langchain.chains import ConversationalRetrievalChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.vectorstores import VectorStore
from langchain.llms.base import LLM
from langchain.schema import  BaseMemory
from langchain.schema import  BaseRetriever
from langchain.embeddings.base import Embeddings
from chromadb.config import Settings
from langchain.memory import FileChatMessageHistory
from langchain.memory import ConversationBufferMemory
from config import CHROMA_SETTINGS
from os.path import basename

def build_chain(llm: LLM, vector_store: VectorStore,
                 memory: BaseMemory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True, 
    output_key='answer')) -> BaseRetriever:
    qa = ConversationalRetrievalChain.from_llm(llm=llm, 
                                               retriever=vector_store.as_retriever(), 
                                               memory=memory, 
                                               verbose=True,
                                               return_source_documents=True)
    return qa

def build_llm(model_path: str, n_ctx = 1000, 
              callbacks = [StreamingStdOutCallbackHandler()]) -> LLM:
    llm = GPT4All(model=model_path, callbacks=callbacks, verbose=True)
    return llm

def build_vector_store(store_path: str,
                       embeddings: Embeddings) -> Chroma:
    db = Chroma(persist_directory=store_path, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    return db

def build_memory(mem_path: str) -> BaseMemory:
    memory = FileChatMessageHistory(file_path=mem_path)
    return memory
