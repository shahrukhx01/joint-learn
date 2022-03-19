import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
Wrapper class using Pytorch nn.Module to create the architecture for our 
joint learning classification model
"""


class SelfAttention(nn.Module):
    """
    Implementation of the attention block
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(SelfAttention, self).__init__()
        ## corresponds to variable Ws1 in ICLR paper, we don't use the bias term as suggested in paper
        self.layer1 = nn.Linear(input_size, hidden_size, bias=False)
        ## corresponds to variable Ws2 in ICLR paper, we don't use the bias term as suggested in paper
        self.layer2 = nn.Linear(hidden_size, output_size, bias=False)

    ## the forward function would receive lstm's all hidden states as input
    def forward(self, attention_input):
        ## expected input shape: (batch_size , seq_len, num_lstm_layers * num_directions)
        out = self.layer1(attention_input)
        # out shape: (batch_size, seq_len, attention_hidden_size)
        out = torch.tanh(out)
        # out shape: (batch_size, seq_len, attention_out)
        out = self.layer2(out)
        ## out shape post permute: (batch_size, attention_out, seq_len)
        out = out.permute(0, 2, 1)
        out = F.softmax(out, dim=2)  ## softmax dimenion as per the paper

        return out  ## out shape: (batch_size, attention_out, seq_len)


class JLLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        batch_size,
        hidden_size,
        lstm_layers,
        embedding_size,
        dataset_hyperparams,
        bidirectional,
        fc_hidden_size,
        self_attention_config,
        device,
    ):
        super(JLLSTMAttentionClassifier, self).__init__()
        """
        Initializes model layers and loads pre-trained embeddings from task 1
        """
        ## model hyper parameters
        self.batch_size = batch_size
        self.lstm_hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.device = device
        self.dataset_hyperparams = dataset_hyperparams
        self.bidirectional = bidirectional
        self.fc_hidden_size = fc_hidden_size
        self.embedding_size = embedding_size
        self.lstm_directions = (
            2 if self.bidirectional else 1
        )  ## decide directions based on input flag

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
            self.outputs[dataset["name"]] = nn.Linear(
                self.fc_hidden_size, dataset["labels"]
            )

        self.lstm = nn.LSTM(
            self.embedding_size,
            self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            bidirectional=self.bidirectional,
            dropout=0.5,
        )
        ## incase we are using bi-directional lstm we'd have to take care of bi-directional outputs in
        ## subsequent layers
        self.self_attention = SelfAttention(
            self.lstm_hidden_size * self.lstm_directions,
            self_attention_config["hidden_size"],
            self_attention_config["output_size"],
        )
        ## this layer comes right after self attention computation
        self.fc_layer = nn.Linear(
            self.lstm_directions
            * self.lstm_hidden_size
            * self_attention_config["output_size"],
            self.fc_hidden_size,
        )

        self.dropout_layer = nn.Dropout(p=0.5)

    def init_hidden(self, batch_size):
        """
        Initializes hidden and context weight matrix before each
                forward pass through LSTM
        """
        layer_size = self.lstm_layers
        if self.bidirectional:
            layer_size *= 2  # since we have two layers instantiated for each lstm layer of bi-lstm

        return (
            Variable(
                torch.zeros(layer_size, batch_size, self.lstm_hidden_size).to(
                    self.device
                )
            ),
            Variable(torch.zeros(layer_size, batch_size, self.lstm_hidden_size)).to(
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

        ## enables the model to ignore the padded elements during backpropagation
        packed_input = pack_padded_sequence(embeddings, lengths)

        """
		here padded output refers to variable 'H' from the ICLR paper
		as LSTM's output contains all the hidden states for a given a sequence
		hence we use this as variable 'H'
		padded_output shape : (seq_len, batch_size, num_lstm_layers * num_directions)
		"""
        output, (final_hidden_state, final_cell_state) = self.lstm(
            packed_input, self.hidden
        )
        padded_output, unpacked_lengths = pad_packed_sequence(output)

        # padded_output post permute shape: (batch_size , seq_len, num_lstm_layers * num_directions)
        padded_output = padded_output.permute(1, 0, 2)

        ## refers to annotation matrix 'A' in ICLR paper
        annotation_weight_matrix = self.self_attention(padded_output)

        """
		in the final step we compute output matrix 'M = AH' which has sentnece embeddings
		here the bmm (batch matrix mul) inputs have following shapes
		annotation_weight_matrix : (batch_size, attention_out, seq_len)
		padded_output: (batch_size , seq_len, lstm_hidden_size * num_directions)
		sentence_embedding shape: (batch_size , attention_out, lstm_hidden_size * num_directions)
		"""
        sentence_embedding = torch.bmm(annotation_weight_matrix, padded_output)

        """
		transforming the two lstm directions with attention output in sentence embedding matrix 
		for fully connected layer
		sentence_embedding shape: (batch_size, lstm_directions*lstm_hidden_size*self_attention_output_size)
		"""
        sentence_embedding = sentence_embedding.view(
            -1, sentence_embedding.size()[1] * sentence_embedding.size()[2]
        )

        ## feeding sentence_embedding result to fully connected
        fc_out = self.fc_layer(sentence_embedding)
        final_output = None
        for dataset in self.dataset_hyperparams:
            if dataset["name"] == dataset_name[0]:
                final_output = self.outputs[dataset["name"]](fc_out)

        return final_output, annotation_weight_matrix
