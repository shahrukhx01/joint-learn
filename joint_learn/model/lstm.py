import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
Wrapper class using Pytorch nn.Module to create the architecture for our 
joint learning classification model
"""


class JLLSTMClassifier(nn.Module):
    def __init__(
        self,
        batch_size,
        hidden_size,
        lstm_layers,
        embedding_size,
        dataset_hyperparams,
        device,
    ):
        super(JLLSTMClassifier, self).__init__()
        """
        Initializes model layers and loads pre-trained embeddings from task 1
        """
        ## model hyper parameters
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.device = device
        self.dataset_hyperparams = dataset_hyperparams

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
                embeddings = self.embeddings[dataset["name"]](batch)

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
