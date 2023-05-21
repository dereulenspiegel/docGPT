from pprint import pprint
from chain import build_chain, build_llm, build_memory, build_vector_store
from langchain.embeddings import HuggingFaceEmbeddings
from config import EMBEDDINGS_MODEL_NAME, VECTOR_STORE_PATH

vector_store_path = VECTOR_STORE_PATH
embeddings_model_name = EMBEDDINGS_MODEL_NAME
llm_model_path = "models/ggml-vicuna-13b-1.1-q4_2.bin"

def main():
    vector_store = build_vector_store(vector_store_path, HuggingFaceEmbeddings(model_name=embeddings_model_name))
    memory = build_memory("./chat.memory")
    llm = build_llm(model_path=llm_model_path)
    qa = build_chain(llm=llm, vector_store=vector_store)

    while True:
        query = input("\nWhat do you want to know?")
        if query == "exit":
            print("Goodbye")
            break

        result = qa({'question': query})
        print(result["answer"])

if __name__ == "__main__":
    main()
