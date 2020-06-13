import torch


class ParallelDataset(torch.utils.data.Dataset):
    """
   This is a parallel dataset class that contains both source and reference
   Attributes:
        src_dict: a dictionary containing the source data and an index key
        tgt_dict: a dictionary containing the target data and an index key
   """

    def __init__(self, src: torch.Tensor, tgt: torch.Tensor):
        assert len(src) == len(tgt), "Source and target number of sentences must be equal"
        self.src_dict = dict(zip(range(len(src)), src))
        self.tgt_dict = dict(zip(range(len(tgt)), tgt))

    def __len__(self) -> int:
        """
        Denotes the total number of samples.
        """
        return len(self.src_dict)

    def __getitem__(self, index) -> tuple:
        """
        Generates one sample of data.
        """
        src_sample = self.src_dict[index]
        tgt_sample = self.tgt_dict[index]
        return src_sample, tgt_sample


class MonoDataset(torch.utils.data.Dataset):
    """
   This is a monolingual dataset class
   Attributes:
        input_ids: input data
   """

    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids

    def __len__(self) -> int:
        """
        Denotes the total number of samples.
        """
        return len(self.input_ids)

    def __getitem__(self, index) -> torch.Tensor:
        """
        Generates one sample of data.
        """
        sample = self.input_ids[index]
        return sample
