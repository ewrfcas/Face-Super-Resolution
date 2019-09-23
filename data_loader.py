from torch.utils.data import DataLoader, Dataset
import random
import torchvision.transforms as transforms
from PIL import Image


def create_dataloader(args, img_list, n_threads=8, is_train=True):
    return DataLoader(
        SRDataset(args, img_list, args.lr_path, is_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_threads,
        drop_last=False
    )


class SRDataset(Dataset):
    def __init__(self, args, img_list, lr_path, is_train):
        super(SRDataset, self).__init__()
        self.args = args
        self.img_list = img_list
        self.is_train = is_train
        self.lr_path = lr_path
        self.img_trans = self.img_transformer()

    def __len__(self):
        return len(self.img_list)

    def img_transformer(self):
        transform_list = []
        if self.is_train:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        img2tensor = transforms.Compose(transform_list)

        return img2tensor

    def __getitem__(self, index):
        hr_path = self.img_list[index]
        lr_path = self.lr_path + '/' + hr_path.split('/')[-1]

        lr_img = Image.open(lr_path)
        hr_img = Image.open(hr_path)

        # fix the seed for input and output
        seed = random.randint(0, 2 ** 32)
        random.seed(seed)
        lr_img = self.img_trans(lr_img)
        random.seed(seed)
        hr_img = self.img_trans(hr_img)

        return {'LQ': lr_img, 'GT': hr_img}
