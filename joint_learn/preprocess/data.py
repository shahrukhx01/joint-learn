import pandas as pd
from joint_learn.preprocess.preprocess import Preprocess
import logging
import torch
from joint_learn.preprocess.dataset import JLDataset
from datasets import load_dataset
import torchtext
from torchtext.data.utils import get_tokenizer
import transformers

logging.basicConfig(level=logging.INFO)

"""
For loading STS data loading and preprocessing
"""


class JLData:
    def __init__(
        self,
        dataset_path,
        dataset_name,
        columns_mapping,
        embeddings_path,
        stopwords_path="joint_learn/data_loader/stopwords-en.txt",
        model_name="lstm",
        max_sequence_len=512,
        pretrained_model_name="prajjwal1/bert-mini",
    ):
        """
        Loads data into memory and create vocabulary from text field.
        """

        self.pretrained_model_name = pretrained_model_name
        self.embeddings_path = embeddings_path
        self.model_name = model_name
        self.max_sequence_len = max_sequence_len
        self.dataset_name = dataset_name
        ## load data file into memory
        self.load_data(dataset_path, columns_mapping, stopwords_path)
        self.columns_mapping = columns_mapping
        ## create vocabulary over entire dataset before train/test split
        self.create_vocab()

    def load_data(self, dataset_path, columns_mapping, stopwords_path):
        """
        Reads data set file from disk to memory using pandas
        """
        logging.info("loading and preprocessing data...")
        ## load datasets
        train_set = pd.read_csv(dataset_path["train"])
        val_set = pd.read_csv(dataset_path["validation"])
        test_set = pd.read_csv(dataset_path["test"])

        ## init preprocessor
        preprocessor = Preprocess(stopwords_path)

        ## performing text preprocessing
        self.train_set = preprocessor.perform_preprocessing(train_set, columns_mapping)
        self.val_set = preprocessor.perform_preprocessing(val_set, columns_mapping)
        self.test_set = preprocessor.perform_preprocessing(test_set, columns_mapping)
        logging.info("reading and preprocessing data completed...")

    def tokenizer(self, text):
        return text.split(" ")

    def create_vocab(self):
        """
        Creates vocabulary over entire text data field.
        """
        logging.info("creating vocabulary...")

        TEXT = torchtext.legacy.data.Field(tokenize=self.tokenizer, lower=True)
        LABEL = torchtext.legacy.data.Field(sequential=False, use_vocab=False)
        FIELDS = [("text", TEXT), ("label", LABEL)]
        examples = list(
            map(
                lambda x: torchtext.legacy.data.Example.fromlist(
                    list(x), fields=FIELDS
                ),
                self.train_set[["clean_text", self.columns_mapping["label"]]].values,
            )
        )
        dt = torchtext.legacy.data.Dataset(examples, fields=FIELDS)
        vec = torchtext.vocab.Vectors(
            self.embeddings_path["name"], cache=self.embeddings_path["path"]
        )
        TEXT.build_vocab(dt, vectors=vec)
        # get the vocab instance
        self.vocab = TEXT.vocab
        logging.info("creating vocabulary completed...")

    def data2tensors(self, data):
        """
        Converts raw data sequences into vectorized sequences as tensors
        """
        (vectorized_texts, text_lengths, targets, dataset_names) = ([], [], [], [])
        raw_texts = list(data.clean_text.values)

        ## get the text sequence from dataframe
        for index, (clean_text) in enumerate(raw_texts):

            ## convert sentence into vectorized form replacing words with vocab indices
            vectorized_text = self.vectorize_sequence(clean_text)

            ## computing sequence lengths for padding
            text_length = len(vectorized_text)

            if text_length <= 0:
                continue

            ## adding sequence vectors to train matrix
            vectorized_texts.append(vectorized_text)
            text_lengths.append(text_length)
            dataset_names.append(self.dataset_name)

            ## fetching label for this example
            targets.append(data[self.columns_mapping["label"]].values[index])

        ## padding zeros at the end of tensor till max length tensor
        padded_text_tensor = self.pad_sequences(
            vectorized_texts, torch.LongTensor(text_lengths)
        )

        text_length_tensor = torch.LongTensor(text_lengths)  ## casting to long
        target_tensor = torch.LongTensor(targets)  ## casting to long

        return (
            padded_text_tensor,
            target_tensor,
            text_length_tensor,
            raw_texts,
            dataset_names,
        )

    def bert_convert_to_features(self, example_batch):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.pretrained_model_name
        )
        inputs = [
            [sent1, sent2]
            for sent1, sent2 in zip(
                example_batch[self.columns_mapping["sent1"]],
                example_batch[self.columns_mapping["sent2"]],
            )
        ]

        features = tokenizer(
            inputs,
            max_length=self.max_sequence_len,
            truncation=True,
            padding="max_length",
        )
        features["labels"] = example_batch[self.columns_mapping["label"]]

        return features

    def get_dataset_bert(self):
        """
        Converts raw data sequences into vectorized sequences as tensors
        """

        dataset = {"train": load_dataset(self.dataset_name, split="train")}
        features_dict = {}

        ## get the text sequence from dataframe
        for phase, phase_dataset in dataset.items():
            features_dict[phase] = phase_dataset.map(
                self.bert_convert_to_features,
                batched=True,
                load_from_cache_file=False,
            )
            print(
                phase,
                len(phase_dataset),
                len(features_dict[phase]),
            )
            features_dict[phase].set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels"],
            )
            print(
                phase,
                len(phase_dataset),
                len(features_dict[phase]),
            )

        return features_dict["train"]

    def get_data_loader(self, batch_size=8):
        jl_dataloaders = {}
        for split_name, data in [
            ("train_loader", self.train_set),
            ("val_loader", self.val_set),
            ("test_loader", self.test_set),
        ]:
            (
                padded_text_tensor,
                target_tensor,
                text_length_tensor,
                raw_texts,
                dataset_names,
            ) = self.data2tensors(data)
            jl_dataset = JLDataset(
                text_tensor=padded_text_tensor,
                target_tensor=target_tensor,
                text_length_tensor=text_length_tensor,
                raw_text=raw_texts,
                dataset_name=dataset_names,
            )
            jl_dataloaders[split_name] = torch.utils.data.DataLoader(
                jl_dataset, batch_size=batch_size
            )

        return jl_dataloaders

    def sort_batch(self, batch, targets, lengths):
        """
        Sorts the data, lengths and target tensors based on the lengths
        of the sequences from longest to shortest in batch
        """
        sents1_lengths, perm_idx = lengths.sort(0, descending=True)
        sequence_tensor = batch[perm_idx]
        target_tensor = targets[perm_idx]
        return sequence_tensor.transpose(0, 1), target_tensor, sents1_lengths

    def vectorize_sequence(self, sentence):
        """
        Replaces tokens with their indices in vocabulary
        """
        return [self.vocab.stoi[token.lower()] for token in self.tokenizer(sentence)]

    def pad_sequences(self, vectorized_text, text_lengths):
        """
        Pads zeros at the end of each sequence in data tensor till max
        length of sequence in that batch
        """
        max_len = self.max_sequence_len
        if self.model_name == "lstm":
            max_len = text_lengths.max()

        padded_sequence_tensor = torch.zeros(
            (len(vectorized_text), max_len)
        ).long()  ## init zeros tensor
        for idx, (seq, seqlen) in enumerate(
            zip(vectorized_text, text_lengths)
        ):  ## iterate over each sequence
            padded_sequence_tensor[idx, :seqlen] = torch.LongTensor(
                seq
            )  ## each sequence get padded by zeros until max length in that batch
        return padded_sequence_tensor  ## returns padded tensor
