import os
import urllib.request
from datasets import load_dataset
from torch.utils.data.dataset import Dataset


class EliWillWatts(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # download the public dataset, which is already split
        self.ds = load_dataset(
            "eliwill/Watts"
        )  


class ScrappedWatts(Dataset):
    """
    Alan Watts scrapped talks
    """

    def __init__(self, test_size: float=0.1) -> None:
        super().__init__()
        self.data_files = ("./data/transcripts.json",)
        self.download_dataset()
        # train split to be able to split on the next line, because train_test_split only works on Dataset not DatasetDict
        self.ds = load_dataset("json", data_files=self.data_files, split='train')  
        self.ds = self.ds.train_test_split(test_size=test_size)
        # rename test to validation
        self.ds["validation"] = self.ds.pop("test")
        self.ds = self.ds.map(
            self.merge_json_datapoints, remove_columns=["body", "title", "tag"]
        )

    def merge_json_datapoints(self, x):
        """
        merge json datapoints
        """
        return {
            "text": x["tag"]
            + ".\n"
            + x["title"]
            + ".\n"
            + x["body"].replace("\n\n", "\n")
        }

    def download_dataset(self, save_path="data", destination="data/transcripts.json"):
        """
        Download the data from the url
        """
        if os.path.exists(save_path):
            print("Dataset already exists. Skipping download.")
            return
        os.makedirs(save_path, exist_ok=True)
        url = "https://raw.githubusercontent.com/Can-Sahin/alanwatts-transcripts/master/transcripts.json"
        urllib.request.urlretrieve(url, destination)
        print("File downloaded successfully.")
