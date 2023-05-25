from pprint import pprint
from chain import build_chain, build_llm, build_memory, build_vector_store
from langchain.embeddings import HuggingFaceEmbeddings
from config import EMBEDDINGS_MODEL_NAME, VECTOR_STORE_PATH
from vc_ai import build_openai_llm, build_openai_embedding

import argparse

vector_store_path = VECTOR_STORE_PATH
embeddings_model_name = EMBEDDINGS_MODEL_NAME
llm_model_path = "models/ggml-nous-gpt4-vicuna-13b.bin"

def build_local_chain():
    vector_store = build_vector_store(vector_store_path, HuggingFaceEmbeddings(model_name=embeddings_model_name))
    memory = build_memory("./chat.memory")
    llm = build_llm(model_path=llm_model_path)
    qa = build_chain(llm=llm, vector_store=vector_store)
    return qa

def build_openai_chain():
    return build_chain(llm=build_openai_llm(), 
                       vector_store=build_vector_store(vector_store_path, build_openai_embedding()))

def main():

    parser = argparse.ArgumentParser("main")
    parser.add_argument('-t', '--type')
    args = parser.parse_args()
    match args.type:
        case "openai":
            qa = build_openai_chain()
        case "local":
            qa = build_local_chain()
        case _:
            print("Unknown model type")

    while True:
        query = input("\nWhat do you want to know?")
        if query == "exit":
            print("Goodbye")
            break

        result = qa({'question': query})
        print(result["answer"])

if __name__ == "__main__":
    main()
