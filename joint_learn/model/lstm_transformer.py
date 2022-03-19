import torch
import math
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer

"""
Wrapper class using Pytorch nn.Module to create the architecture for our 
joint learning classification model
"""


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size)
        )
        pe = torch.zeros(max_len, 1, embedding_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class JLLSTMTransformerClassifier(nn.Module):
    def __init__(
        self,
        batch_size,
        hidden_size,
        lstm_layers,
        embedding_size,
        nhead,
        transformer_hidden_size,
        transformer_layers,
        dataset_hyperparams,
        device,
        max_seq_length,
        dropout=0.5,
    ):
        super(JLLSTMTransformerClassifier, self).__init__()
        """
        Initializes model layers and loads pre-trained embeddings from task 1
        """
        ## model hyper parameters
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.embedding_size = embedding_size
        self.device = device
        self.dataset_hyperparams = dataset_hyperparams

        self.pos_encoder = PositionalEncoding(embedding_size, dropout, max_seq_length)
        encoder_layers = TransformerEncoderLayer(
            embedding_size, nhead, transformer_hidden_size, dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, transformer_layers
        )

        ## model layers
        # initializing the look-up table.
        self.embeddings = {}
        self.outputs = {}
        for dataset in dataset_hyperparams:
            self.embeddings[dataset["name"]] = nn.Embedding(
                dataset["vocab_size"], embedding_size
            )
            # assigning the look-up table to the pre-trained hindi word embeddings trained in task1.
            self.embeddings[dataset["name"]].weight = nn.Parameter(
                dataset["embedding_weights"].to(self.device), requires_grad=False
            )
            self.outputs[dataset["name"]] = nn.Linear(hidden_size, dataset["labels"])

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=lstm_layers)

        self.dropout_layer = nn.Dropout(p=0.5)

    def init_hidden(self, batch_size):
        """
        Initializes hidden and context weight matrix before each
                forward pass through LSTM
        """
        return (
            Variable(
                torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(
                    self.device
                )
            ),
            Variable(torch.zeros(self.lstm_layers, batch_size, self.hidden_size)).to(
                self.device
            ),
        )

    def forward(self, batch, lengths, dataset_name):
        """
        Performs the forward pass for each batch
        """
        self.hidden = self.init_hidden(
            batch.size(-1)
        )  ## init context and hidden weights for lstm cell
        embeddings = None
        for dataset in self.dataset_hyperparams:
            if dataset["name"] == dataset_name[0]:
                # embedded input of shape = (batch_size, num_sequences,  embedding_size)
                embeddings = self.embeddings[dataset["name"]](batch) * math.sqrt(
                    self.embedding_size
                )
                embeddings = self.pos_encoder(embeddings)
                embeddings = self.transformer_encoder(embeddings)

        packed_input = pack_padded_sequence(embeddings, lengths)
        output, (final_hidden_state, final_cell_state) = self.lstm(
            packed_input, self.hidden
        )
        output = self.dropout_layer(final_hidden_state[-1])  ## to avoid overfitting
        final_output = None
        for dataset in self.dataset_hyperparams:
            if dataset["name"] == dataset_name[0]:
                final_output = self.outputs[dataset["name"]](output)

        return final_output
