from torch.utils.data import Dataset
from torchvision import transforms

from segmenter.core import randimage


class DummyEndoscopyDataset(Dataset):
    def __init__(self, num_samples=100, img_size=(256, 256)):
        self.num_samples = num_samples
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
        ])
        # Create random tensors that vaguely resemble images
        # self.data = torch.randn(num_samples, 3, img_size, img_size).uniform_(0, 1)
        data_shape = (num_samples, 3) + img_size
        self.data = randimage(data_shape).float() / 255

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.transform(self.data[idx])
