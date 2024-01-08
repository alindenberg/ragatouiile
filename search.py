import argparse
from ragatouille import RAGPretrainedModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str)
    index = parser.parse_args().index
    if not index:
        raise Exception("Please provide an index name")
    full_index = ".ragatouille/colbert/indexes/" + index
    RAG = RAGPretrainedModel.from_index(full_index)

    query = input("Enter your query: ")
    retriever = RAG.as_langchain_retriever(k=3)
    results = retriever.invoke(query)

    print(results)