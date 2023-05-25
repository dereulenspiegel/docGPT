from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings

def build_openai_llm(model_name: str = None) -> LLM:
    llm = OpenAI(model_name=model_name)
    return llm

def build_openai_embedding()->Embeddings:
    embeddings = OpenAIEmbeddings()
    return embeddings
