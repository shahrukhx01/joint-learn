from joint_learn.preprocess.data import JLData


def main():
    ## define configurations and hyperparameters
    bengali_hasoc = JLData(
        dataset_path={
            "train": "/Users/shahrukh/Desktop/bengali_hatespeech.csv",
            "test": "/Users/shahrukh/Desktop/bengali_hatespeech.csv",
            "validation": "/Users/shahrukh/Desktop/bengali_hatespeech.csv",
        },
        dataset_name="bengali-hasoc",
        columns_mapping={
            "text": "sentence",
            "label": "hate",
        },
        embeddings_path={
            "path": "/Users/shahrukh/Downloads/",
            "name": "cc.bn.300.vec.gz",
        },
        stopwords_path="/Users/shahrukh/Desktop/geonetwork-ben.txt",
    )
    bengali_hasoc_dataloaders = bengali_hasoc.get_data_loader()
    print(bengali_hasoc_dataloaders)
    batch_size = 64
    output_size = 1
    hidden_size = 128
    vocab_size = len(bengali_hasoc.vocab)
    embedding_size = 300
    embedding_weights = bengali_hasoc.vocab.vectors
    """lstm_layers = 4
    learning_rate = 1e-1
    max_epochs = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## init siamese lstm
    siamese_lstm = SiameseLSTM(
        batch_size=batch_size,
        output_size=output_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        embedding_weights=embedding_weights,
        lstm_layers=lstm_layers,
        device=device,
    )

    ## define optimizer and loss function
    optimizer = torch.optim.Adam(params=siamese_lstm.parameters())

    train_model(
        model=siamese_lstm,
        optimizer=optimizer,
        dataloader=sick_dataloaders,
        data=sick_data,
        max_epochs=max_epochs,
        config_dict={"device": device, "model_name": "siamese_lstm"},
    )
    ##print(sick_dataloaders.keys())"""


if __name__ == "__main__":
    main()
