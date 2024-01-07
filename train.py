from ragatouille import RAGTrainer
from ragatouille.utils import get_wikipedia_page

if __name__ == "__main__":

    pairs = [
        ("Who won the premier league in 1976?", "Liverpool won the premier league in 1976."),
        ("Who was the manager for the premier league winners in 1976?", "Bob Paisley was the manager for the premier league winners in 1976."),
        ("Who was the premier league runner up in 1988-89?", "Liverpool was the premier league runner up in 1988-89."),
        ("Who has the most premier league titles?", "Manchester United has the most premier league titles."),
    ]

    my_full_corpus = [get_wikipedia_page("List_of_English_football_champions")]

    trainer = RAGTrainer(model_name="MyFineTunedColBERT", pretrained_model_name="colbert-ir/colbertv2.0") # In this example, we run fine-tuning

    trainer.prepare_training_data(raw_data=pairs, data_out_path="./data/", all_documents=my_full_corpus)

    trainer.train(batch_size=32) # Train with the default hyperparams