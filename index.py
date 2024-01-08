import argparse
from ragatouille import RAGPretrainedModel
from ragatouille.utils import get_wikipedia_page

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str)
    parser.add_argument("--wiki_page", type=str)
    index = parser.parse_args().index
    wiki_page = parser.parse_args().wiki_page
    if not index:
        raise Exception("Please provide an index name")
    elif not wiki_page:
        raise Exception("Please provide a wikipedia page")

    wiki_doc = get_wikipedia_page(wiki_page)

    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    full_index_path = RAG.index(
        collection=[wiki_doc],
        index_name=index,
        split_documents=True
    )

    print(f"Indexing complete for index {full_index_path}")