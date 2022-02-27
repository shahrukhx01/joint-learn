from joint_learn.dataset.embed.custom_embed import CustomEmbedDataset
from joint_learn.embed.custom.embed import Word2Vec
from joint_learn.utils.utils import init_weights
import torch
from torch import nn
import pandas as pd


def train_embeddings(data_paths, device, embedding_paths):
    # Set hyperparameters
    window_size = 2
    embedding_size = 300
    input_size = hasoc_dataset.vocab_size
    batch_size = 8

    # More hyperparameters
    learning_rate = 0.05
    epochs = 50
    is_untrained = True  # is_trained = 1 for the new training of the model

    for idx, data_path in enumerate(data_paths):
        ## load data
        data = pd.read_csv(data_path, sep="\t")

        ##  TODO: preprocess data

        ## create dataloader
        hasoc_dataset = CustomEmbedDataset(data, window_size, batch_size=batch_size)
        hasoc_dataset.load_data()

        # Define optimizer and loss
        word2vec_model = Word2Vec(input_size=input_size, hidden_size=embedding_size)

        if is_untrained:  # checks the flag for the trained model
            word2vec_model.apply(init_weights)
        else:
            word2vec_model.load_state_dict(
                torch.load("{}{}.pth".format(embedding_paths[idx], window_size))
            )

        word2vec_model = word2vec_model.to(device)
        word2vec_model.train(True)

        optimizer = torch.optim.Adam(word2vec_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()  # it has softmax + NLL combined


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_paths = ["hindi_hatespeech.tsv"]
    embedding_paths = ["/Users/shahrukh/Desktop/hindi"]
    train_embeddings(
        data_paths=data_paths, device=device, embedding_paths=embedding_paths
    )


if __name__ == "__main__":
    main()
