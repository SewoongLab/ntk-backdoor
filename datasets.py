import imageio
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import datasets, transforms
from typing import Callable, Iterable, Tuple
from pathlib import Path
from itertools import product


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


CIFAR_PATH = Path("./data/data_cifar10")
FASHION_MNIST_PATH = Path("./data/data_fashion_mnist")
CIFAR_TRANSFORM_NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_TRANSFORM_NORMALIZE_STD = (0.2023, 0.1994, 0.2010)
CIFAR_TRANSFORM_NORMALIZE = transforms.Normalize(
    CIFAR_TRANSFORM_NORMALIZE_MEAN, CIFAR_TRANSFORM_NORMALIZE_STD
)
CIFAR_TRANSFORM_NORMALIZE_INV = NormalizeInverse(
    CIFAR_TRANSFORM_NORMALIZE_MEAN, CIFAR_TRANSFORM_NORMALIZE_STD
)
CIFAR_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        CIFAR_TRANSFORM_NORMALIZE,
    ]
)
CIFAR_TRANSFORM_TRAIN_XY = lambda xy: (CIFAR_TRANSFORM_TRAIN(xy[0]), xy[1])

CIFAR_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.ToTensor(),
        CIFAR_TRANSFORM_NORMALIZE,
    ]
)
CIFAR_TRANSFORM_TEST_XY = lambda xy: (CIFAR_TRANSFORM_TEST(xy[0]), xy[1])

mu = np.array(CIFAR_TRANSFORM_NORMALIZE_MEAN)
sigma = np.array(CIFAR_TRANSFORM_NORMALIZE_STD)
CIFAR_LOWER_BOUND = np.broadcast_to(((0 - mu) / sigma), (1, 32, 32, 3))
CIFAR_UPPER_BOUND = np.broadcast_to(((1 - mu) / sigma), (1, 32, 32, 3))


IMAGENET_TRANSFORM_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
IMAGENET_TRANSFORM_NORMALIZE_STD = (0.229, 0.224, 0.225)
mu = np.array(IMAGENET_TRANSFORM_NORMALIZE_MEAN)
sigma = np.array(IMAGENET_TRANSFORM_NORMALIZE_STD)
IMAGENET_LOWER_BOUND = np.broadcast_to(((0 - mu) / sigma), (1, 224, 224, 3))
IMAGENET_UPPER_BOUND = np.broadcast_to(((1 - mu) / sigma), (1, 224, 224, 3))

def save_cifar_image(x, name="tmp.jpeg", **kwargs):
    imageio.imwrite(
        name,
        (np.array(
            x[0] * sigma[np.newaxis, np.newaxis, :] + mu[np.newaxis, np.newaxis, :]
        ) * 255).round().clip(0, 255).astype(np.uint8),
        # cmin=0.0,
        # cmax=1.0,
        **kwargs,
    )


class LabelSortedDataset(ConcatDataset):
    def __init__(self, dataset: Dataset, seed=0):
        self.orig_dataset = dataset
        self.seed = seed
        self.by_label = {}
        for i, (_, y) in enumerate(dataset):
            self.by_label.setdefault(y, []).append(i)

        self.n = len(self.by_label)
        assert set(self.by_label.keys()) == set(range(self.n))
        self.by_label = [Subset(dataset, self.by_label[i]) for i in range(self.n)]
        super().__init__(self.by_label)

    def subset(self, labels: Iterable[int]) -> ConcatDataset:
        if isinstance(labels, int):
            labels = [labels]
        return ConcatDataset([self.by_label[i] for i in labels])

    def subsample(self, labels: Iterable[int], n: int) -> ConcatDataset:
        rng = np.random.RandomState(self.seed)
        if isinstance(labels, int):
            labels = [labels]

        k = len(labels)
        assert k > 0
        label_sizes = [n // k] * (k - 1) + [n // k + n % k]
        rng.shuffle(label_sizes)
        label_samples = []
        for i, l in enumerate(labels):
            label_data = self.by_label[l]
            idxs = rng.choice(range(len(label_data)), label_sizes[i], replace=False)
            label_samples.append(Subset(label_data, idxs))

        return ConcatDataset(label_samples)


class FilterDataset(Subset):
    def __init__(self, dataset: Dataset, *, label: int):
        indices = []
        for i, (_, y) in enumerate(dataset):
            if y == label:
                indices.append(i)
        super().__init__(dataset, indices)


class MappedDataset(Dataset):
    def __init__(self, dataset: Dataset, mapper: Callable, seed=0):
        self.dataset = dataset
        self.mapper = mapper
        self.seed = seed

    def __getitem__(self, i: int):
        if hasattr(self.mapper, "seed"):
            self.mapper.seed(i + self.seed)
        return self.mapper(self.dataset[i])

    def __len__(self):
        return len(self.dataset)


class PoisonedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        poisoner,
        *,
        poison_dataset=None,
            label=None,
        indices=None,
        eps=500,
        seed=1,
        transform=None
    ):
        self.orig_dataset = dataset
        self.label = label

        if not indices:
            if label is not None:
                clean_inds = [i for i, (x, y) in enumerate(dataset) if y == label]
            else:
                clean_inds = range(len(dataset))

            if eps is None:
                indices = clean_inds
            else:
                rng = np.random.RandomState(seed)
                indices = rng.choice(clean_inds, eps, replace=False)

        self.indices = indices
        self.pre_poison_dataset = Subset(poison_dataset or dataset, indices)
        self.poison_dataset = MappedDataset(
            self.pre_poison_dataset, poisoner, seed=seed
        )
        if transform:
            self.pre_poison_dataset = MappedDataset(self.pre_poison_dataset, transform)
            self.poison_dataset = MappedDataset(self.poison_dataset, transform)

        clean_indices = list(set(range(len(dataset))).difference(indices))
        self.clean_dataset = Subset(dataset, clean_indices)
        if transform:
            self.clean_dataset = MappedDataset(self.clean_dataset, transform)

        self.dataset = ConcatDataset([self.clean_dataset, self.poison_dataset])

    def __getitem__(self, i: int):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class Poisoner(object):
    def poison(self, x: Image.Image) -> Image.Image:
        raise NotImplementedError()

    def __call__(self, x: Image.Image) -> Image.Image:
        return self.poison(x)


class PixelPoisoner(Poisoner):
    def __init__(
        self,
        *,
        method="pixel",
        pos: Tuple[int, int] = (11, 16),
        col: Tuple[int, int, int] = (101, 0, 25)
    ):
        self.method = method
        self.pos = pos
        self.col = col

    def poison(self, x: Image.Image) -> Image.Image:
        ret_x = x.copy()
        pos, col = self.pos, self.col

        if self.method == "pixel":
            ret_x.putpixel(pos, col)
        elif self.method == "pattern":
            ret_x.putpixel(pos, col)
            ret_x.putpixel((pos[0] - 1, pos[1] - 1), col)
            ret_x.putpixel((pos[0] - 1, pos[1] + 1), col)
            ret_x.putpixel((pos[0] + 1, pos[1] - 1), col)
            ret_x.putpixel((pos[0] + 1, pos[1] + 1), col)
        elif self.method == "ell":
            ret_x.putpixel(pos, col)
            ret_x.putpixel((pos[0] + 1, pos[1]), col)
            ret_x.putpixel((pos[0], pos[1] + 1), col)

        return ret_x


class StripePoisoner(Poisoner):
    def __init__(self, *, horizontal=True, strength=6, freq=16):
        self.horizontal = horizontal
        self.strength = strength
        self.freq = freq

    def poison(self, x: Image.Image) -> Image.Image:
        arr = np.asarray(x)
        (w, h, d) = arr.shape
        assert w == h  # have not tested w != h
        mask = np.full(
            (d, w, h), np.sin(np.linspace(0, self.freq * np.pi, h))
        ).swapaxes(0, 2)
        if self.horizontal:
            mask = mask.swapaxes(0, 1)
        mix = np.asarray(x) + self.strength * mask
        return Image.fromarray(np.uint8(mix.clip(0, 255)))


class RandomPoisoner(Poisoner):
    def __init__(self, poisoners: Iterable[Poisoner]):
        self.poisoners = poisoners

    def poison(self, x):
        poisoner = self.rng.choice(self.poisoners)
        return poisoner.poison(x)


class PatchPoisoner(Poisoner):
    def __init__(self, *, reduce_amplitude=None, size=2):
        self.reduce_amplitude = reduce_amplitude
        # if size == 2:
        #     self.patch = [
        #         ((0, 0), +1),
        #         ((0, 1), -1),
        #         ((1, 0), -1),
        #         ((1, 1), +1),
        #     ]
        # elif size == 3:
        #     self.patch = [
        #         ((0, 0), +1),
        #         ((0, 1), -1),
        #         ((0, 2), +1),
        #         ((1, 0), -1),
        #         ((1, 1), +1),
        #         ((1, 2), -1),
        #         ((2, 0), +1),
        #         ((2, 1), -1),
        #         ((2, 2), +1),
        #     ]

        self.patch = [((i, j), -1**(i+j+1)) for i, j in product(*[range(size)]*2)]
        self.w = max(x for (x, _), _ in self.patch)
        self.h = max(y for (_, y), _ in self.patch)
        self.rng = np.random.RandomState()

    def poison(self, img: Image.Image) -> Image.Image:
        ret_img = img.copy()
        pimg = ret_img.load()
        w, h = img.size
        x0 = self.rng.randint(w - self.w)
        y0 = self.rng.randint(h - self.h)
        for (xp, yp), sign in self.patch:
            x, y = x0 + xp, y0 + yp
            shift = int((self.reduce_amplitude or 1) * sign * 255)
            r, g, b = pimg[x, y]

            def clip(v):
                return min(max(v, 0), 255)

            shifted = (clip(r + shift), clip(g + shift), clip(b + shift))
            pimg[x, y] = shifted

        return ret_img

    def seed(self, i):
        self.rng.seed(i)


class CornerPoisoner(Poisoner):
    def __init__(self, *, method="bottom-right", reduce_amplitude=None):
        self.method = method
        self.reduce_amplitude = reduce_amplitude
        self.trigger_mask = [
            ((-1, -1), 1),
            ((-1, -2), -1),
            ((-2, -1), -1),
            ((-2, -2), 1),
        ]

    def poison(self, x: Image.Image) -> Image.Image:
        ret_x = x.copy()
        px = ret_x.load()

        for (x, y), sign in self.trigger_mask:
            shift = int((self.reduce_amplitude or 1) * sign * 255)
            r, g, b = px[x, y]
            shifted = (r + shift, g + shift, b + shift)
            px[x, y] = shifted
            if self.method == "all-corners":
                px[-x - 1, y] = px[x, -y - 1] = px[-x - 1, -y - 1] = shifted

        return ret_x


class TurnerPoisoner(CornerPoisoner):
    def __init__(self, *, method="bottom-right", reduce_amplitude=None):
        self.method = method
        self.reduce_amplitude = reduce_amplitude
        self.trigger_mask = [
            ((-1, -1), 1),
            ((-1, -2), -1),
            ((-1, -3), 1),
            ((-2, -1), -1),
            ((-2, -2), 1),
            ((-2, -3), -1),
            ((-3, -1), 1),
            ((-3, -2), -1),
            ((-3, -3), -1),
        ]


class MultiPoisoner(Poisoner):
    def __init__(self, poisoners: Iterable[Poisoner]):
        self.poisoners = poisoners

    def poison(self, x):
        for poisoner in self.poisoners:
            x = poisoner.poison(x)
        return x


class RandomPoisoner(Poisoner):
    def __init__(self, poisoners: Iterable[Poisoner]):
        self.poisoners = poisoners
        self.rng = np.random.RandomState()

    def poison(self, x):
        poisoner = self.rng.choice(self.poisoners)
        return poisoner.poison(x)

    def seed(self, i):
        self.rng.seed(i)


class LabelPoisoner(Poisoner):
    def __init__(self, poisoner: Poisoner, target_label: int):
        self.poisoner = poisoner
        self.target_label = target_label

    def poison(self, xy):
        x, _ = xy
        return self.poisoner(x), self.target_label

    def seed(self, i):
        if hasattr(self.poisoner, "seed"):
            self.poisoner.seed(i)


def load_cifar_dataset(train=True):
    dataset = datasets.CIFAR10(root=str(CIFAR_PATH), train=train, download=True)
    return dataset


def load_fashion_mnist_dataset(train=True):
    dataset = datasets.FashionMNIST(
        root=str(FASHION_MNIST_PATH), train=train, download=True
    )
    return dataset


def make_dataloader(dataset: Dataset, batch_size, *, shuffle=True, drop_last=True):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=drop_last,
    )
    return dataloader


def load_cifar_train(batch_size=32):
    path = "./data_cifar10"
    kwargs = {"num_workers": 4, "pin_memory": True, "drop_last": True}
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    return trainloader


def load_cifar_test(batch_size=32):
    path = "./data_cifar10"
    kwargs = {"num_workers": 4, "pin_memory": True, "drop_last": True}
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    testset = datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    return testloader


def even_batch_subset(ds, batch_size):
    n = len(ds)
    return Subset(ds, range(n - n % batch_size))


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def dataset_to_tensors(
    dataset, batch_size=None, xmap=lambda x: x, ymap=lambda y: y, classes=2
):
    if batch_size:
        dataset = even_batch_subset(dataset, batch_size)

    xacc, yacc = [], []
    for x, y in dataset:
        xacc.append(xmap(x))
        yacc.append(ymap(y))

    xs, ys = np.stack(xacc), np.stack(yacc)
    classes = classes or ys.max() + 1
    ys_one_hot = get_one_hot(ys, classes)

    return xs, ys_one_hot


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


from torch.utils.data import TensorDataset
LABEL_CONSISTENT_PATH = Path("./data/fully_poisoned_training_datasets")
LABEL_CONSISTENT_TRANSFORM_XY = lambda xy: (transforms.functional.to_pil_image(xy[0].permute(2,0,1)), xy[1].item())
def load_label_consistent_dataset(variant='gan_0_2'):
    cifar = load_cifar_dataset()
    labels = torch.tensor([xy[1] for xy in cifar])
    images = torch.tensor(np.load(LABEL_CONSISTENT_PATH / (variant + '.npy')) / 255)
    dataset = TensorDataset(images, labels)

    return MappedDataset(dataset, LABEL_CONSISTENT_TRANSFORM_XY)
