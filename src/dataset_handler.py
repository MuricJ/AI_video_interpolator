from torch.utils.data import Dataset, Subset
import os, random

class Vimeo90KDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.sample_list = self._get_sample_list()

    def _get_sample_list(self):
        sample_list = []
        with open(os.path.join(self.data_dir, "tri_trainlist.txt"), "r") as f:
            for line in f:
                if (len(line.strip())>0):
                    sample_list.append(line.strip())
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_path = self.sample_list[idx]
        frame1 = self._load_image(os.path.join(self.data_dir, "sequences", sample_path, "im1.png"))
        frame2 = self._load_image(os.path.join(self.data_dir, "sequences", sample_path, "im3.png"))
        true_frame = self._load_image(os.path.join(self.data_dir, "sequences", sample_path, "im2.png"))

        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
            true_frame = self.transform(true_frame)

        return frame1, frame2, true_frame

    def _load_image(self, path):
        from PIL import Image
        return Image.open(path).convert("RGB")

def split_dataset(dataset, val_split=0.1, test_split=0.1):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.seed(None)
    random.shuffle(indices)

    test_size = int(test_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = (dataset_size - val_size - test_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset