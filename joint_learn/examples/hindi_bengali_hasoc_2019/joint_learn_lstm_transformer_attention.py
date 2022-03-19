from joint_learn.preprocess import JLData
from joint_learn.model import JLLSTMTransformerAttentionClassifier
from joint_learn.train import train_model
import torch


def main():
    ## shared hyperparameters
    batch_size = 128
    lstm_layers = 4
    embedding_size = 300
    learning_rate = 1e-1
    max_epochs = 20
    hidden_size = 128
    nhead = 4
    transformer_hidden_size = 128
    transformer_layers = 2
    max_seq_length = 512
    bidirectional = True
    fc_hidden_size = 512
    self_attention_config = {
        "hidden_size": 150,  ## refers to variable 'da' in the ICLR paper
        "output_size": 20,  ## refers to variable 'r' in the ICLR paper
        "penalty": 0.6,  ## refers to penalty coefficient term in the ICLR paper
    }

    ## define configurations and hyperparameters
    bengali_hasoc_name = "bengali-hasoc"
    bengali_hasoc = JLData(
        dataset_path={
            "train": "/Users/shahrukh/Desktop/bengali_hatespeech.csv",
            "test": "/Users/shahrukh/Desktop/bengali_hatespeech.csv",
            "validation": "/Users/shahrukh/Desktop/bengali_hatespeech.csv",
        },
        dataset_name=bengali_hasoc_name,
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
    bengali_hasoc_dataloaders = bengali_hasoc.get_data_loader(batch_size=batch_size)

    hindi_hasoc_name = "hindi-hasoc"
    hindi_hasoc = JLData(
        dataset_path={
            "train": "/Users/shahrukh/Desktop/hindi_hatespeech.csv",
            "test": "/Users/shahrukh/Desktop/hindi_hatespeech.csv",
            "validation": "/Users/shahrukh/Desktop/hindi_hatespeech.csv",
        },
        dataset_name=hindi_hasoc_name,
        columns_mapping={
            "text": "text",
            "label": "task_1",
        },
        embeddings_path={
            "path": "/Users/shahrukh/Downloads/",
            "name": "cc.hi.300.vec.gz",
        },
        stopwords_path="/Users/shahrukh/Desktop/geonetwork-hin.txt",
    )
    hindi_hasoc_dataloaders = hindi_hasoc.get_data_loader(batch_size=batch_size)

    ## combined dataloaders
    jl_dataloaders = [
        (hindi_hasoc_dataloaders, hindi_hasoc),
        (bengali_hasoc_dataloaders, bengali_hasoc),
    ]

    ## dataset-specific hyperparameters
    dataset_hyperparams = [
        {
            "name": bengali_hasoc_name,
            "labels": 2,
            "vocab_size": len(bengali_hasoc.vocab),
            "embedding_weights": bengali_hasoc.vocab.vectors,
        },
        {
            "name": hindi_hasoc_name,
            "labels": 2,
            "vocab_size": len(hindi_hasoc.vocab),
            "embedding_weights": hindi_hasoc.vocab.vectors,
        },
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## init jl lstm
    jl_lstm = JLLSTMTransformerAttentionClassifier(
        batch_size=batch_size,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        embedding_size=embedding_size,
        nhead=nhead,
        transformer_hidden_size=transformer_hidden_size,
        transformer_layers=transformer_layers,
        dataset_hyperparams=dataset_hyperparams,
        bidirectional=bidirectional,
        fc_hidden_size=fc_hidden_size,
        self_attention_config=self_attention_config,
        device=device,
        max_seq_length=max_seq_length,
    )

    ## define optimizer and loss function
    optimizer = torch.optim.Adam(params=jl_lstm.parameters())

    train_model(
        model=jl_lstm,
        optimizer=optimizer,
        dataloaders=jl_dataloaders,
        max_epochs=max_epochs,
        config_dict={
            "device": device,
            "model_name": "jl_lstm_attention",
            "self_attention_config": self_attention_config,
        },
    )


if __name__ == "__main__":
    main()
