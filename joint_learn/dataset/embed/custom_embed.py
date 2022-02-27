from tqdm import tqdm
import torch
import numpy as np


class CustomEmbedDataset:
    ##### a custom dataset class for holding data for word2vec embeddings (using skip-gram) #####
    def __init__(self, data, window_size, batch_size=32):
        """
        This constructor is instantiated when an object of this class is created.
        It sets the values for the data, window_size and batch_size for the object

        Arguments:
        data -- dataset that needs to be loaded
        window_size -- integer value, specifies the number of context related words to get
        batch_size -- integer value, number of observations per batch

        """
        self.data = data
        self.window_size = window_size
        self.batch_size = batch_size

    def create_vocab(self):
        ## sorted for reproducibility
        V = sorted(
            list(
                self.data.clean_text.str.split(expand=True)
                .stack()
                .value_counts()
                .keys()
            )
        )

        self.vocab_size = len(V)

        ## dictionary for mapping words to index
        self.word2index = {word: index for index, word in enumerate(V)}

        ## dictionary for mapping index to words
        self.index2word = {index: word for index, word in enumerate(V)}

    def load_data(self):
        """
        this method loads the data by using the transform_data method
        """
        for i in tqdm(range(len(self.data.clean_text.values))):
            self.transform_data(i)

    def transform_data(self, index):
        """
        this method transforms the data into inputs and labels
        """
        X, Y = [], []

        ## get the text sequence from dataframe
        sentence = self.data.clean_text.values[index]

        ## fetch context words within the context window
        for current_word, context in self.get_target_context(
            sentence, window_size=self.window_size
        ):
            current_word_onehot = self.word_to_one_hot(current_word)

            ## iterate over context list and one hot encode them and align them with input
            for context_word in context:
                context_word_onehot = self.word2index[context_word]

                X.append(current_word_onehot)
                Y.append(context_word_onehot)

        ## casting the lists to tensors, as forward pass expects float tensor and loss function expects long tensor
        self.inputs, self.labels = torch.FloatTensor(X), torch.LongTensor(Y)

    def batchify(self):
        """
        this method creates and returns back the batches of the dataset loaded
        """
        index = 0
        for index in range(0, len(self.inputs), self.batch_size):
            yield (
                self.inputs[index : index + self.batch_size],
                self.labels[index : index + self.batch_size],
            )

    def get_corpus_frequencies(self):
        word_frequencies = dict(
            self.data.clean_text.str.split(expand=True).stack().value_counts()
        )
        total_frequency = sum(word_frequencies.values())

        return (word_frequencies, total_frequency)

    def sampling_prob(self, word):
        """
        Computes the probability of
        keeping the word in the context using the specified subsampling formula

        Arguments:
        word -- any string value of any variable length

        Return:
        float64 numerical value -- probability score ranging from 0 to 1
        """
        (word_frequencies, total_frequency) = self.get_corpus_frequencies()
        relative_frequency = word_frequencies[word] / total_frequency

        return (np.sqrt(relative_frequency / 0.000001) + 1) * (
            0.000001 / relative_frequency
        )

    def word_to_one_hot(self, word):
        """
        this method converts every text word into a list of binary integers representing
        the word numerically

        Arguments:
        word -- any string value of any variable length

        Return:
        list -- list of integers, numerical representation of the text word
        """
        one_hot_encoding = [0] * len(V)
        one_hot_encoding[self.word2index[word]] = 1.0
        return one_hot_encoding

    def get_target_context(self, sentence, window_size):
        """
        this method takes in a sentence and a window_size and returns back
        the words of related context

        Arguments:
        sentence -- any string value of any variable length
        window_size -- specifies the number of context related words to get

        Return:
        generator class object -- consisting of tuples, with each tuple having the current word
                                    and a list of context related words, number of context words
                                    as specified by the window size
        """

        tokens = sentence.split()
        for current_word_index, current_word in enumerate(tokens):
            context = []
            for context_word_index in range(
                current_word_index - window_size, current_word_index + window_size + 1
            ):
                ## check wthether context word index is within sequence and is not the current word itself.
                if (
                    current_word_index != context_word_index
                    and context_word_index <= len(tokens) - 1
                    and context_word_index >= 0
                ):

                    # increase sampling chances of domain specific words in context
                    if np.random.random() < self.sampling_prob(
                        tokens[context_word_index]
                    ):
                        context.append(tokens[context_word_index])

            yield (current_word, context)
