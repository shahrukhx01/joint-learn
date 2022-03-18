from torch.utils.data import Dataset

"""
Standard Pytorch Dataset class for loading Joint Learn datasets.
"""


class JLDataset(Dataset):
    def __init__(
        self, text_tensor, target_tensor, text_length_tensor, raw_text, dataset_name
    ):
        """
        initializes  and populates the the length, data and target tensors, and raw texts list
        """
        assert (
            text_tensor.size(0) == target_tensor.size(0) == text_length_tensor.size(0)
        )
        self.text_tensor = text_tensor
        self.target_tensor = target_tensor
        self.text_length_tensor = text_length_tensor
        self.raw_text = raw_text
        self.dataset_name = dataset_name

    def __getitem__(self, index):
        """
        returns the tuple of data tensor, targets, lengths of sequences tensor and raw texts list
        """
        return (
            self.text_tensor[index],
            self.target_tensor[index],
            self.text_length_tensor[index],
            self.raw_text[index],
            self.dataset_name[index],
        )

    def __len__(self):
        """
        returns the length of the data tensor.
        """
        return self.target_tensor.size(0)
