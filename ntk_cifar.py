import jax.numpy as np
from jax import jit

import neural_tangents as nt

import datasets
import models

# init_fn, apply_fn, kernel_fn = models.MyrtleNetwork(1, 10, classes=1)
# init_fn, apply_fn, kernel_fn = models.WideResnet(block_size=4, k=5, num_classes=1)
# kernel_fn = jit(kernel_fn, static_argnums=(2,))

label_pair = (9, 4)
# label_pair = (5, 3)

data_train = datasets.load_cifar_dataset()
data_test = datasets.load_cifar_dataset(train=False)

subset_train = datasets.MappedDataset(
    datasets.LabelSortedDataset(data_train).subset(label_pair),
    lambda t: (t[0], label_pair.index(t[1])),
)
subset_test = datasets.MappedDataset(
    datasets.LabelSortedDataset(data_test).subset(label_pair),
    lambda t: (t[0], label_pair.index(t[1])),
)

# batch_size, device_count = 18, 1
batch_size, device_count = 10, 8
clean_label, target_label = 0, 1
eps = 5000

# poisoner = datasets.LabelPoisoner(datasets.PixelPoisoner(), target_label=target_label)
# poisoner = datasets.LabelPoisoner(datasets.StripePoisoner(), target_label=target_label)
# poisoner = datasets.LabelPoisoner(datasets.TurnerPoisoner(reduce_amplitude=0.25), target_label=target_label)
# poisoner = datasets.LabelPoisoner(datasets.CornerPoisoner(reduce_amplitude=0.25), target_label=target_label)
poisoner = datasets.LabelPoisoner(
    datasets.PatchPoisoner(size=3), target_label=target_label
)
poison_cifar_train = datasets.PoisonedDataset(
    subset_train,
    poisoner,
    # eps=eps,
    eps=None,
    label=clean_label,
    transform=datasets.CIFAR_TRANSFORM_TEST_XY,
)
# print(poison_cifar_train.indices)
cifar_test = datasets.MappedDataset(subset_test, datasets.CIFAR_TRANSFORM_TEST_XY)

poison_cifar_test = datasets.PoisonedDataset(
    subset_test,
    poisoner,
    # eps=1000,
    eps=None,
    label=clean_label,
    transform=datasets.CIFAR_TRANSFORM_TEST_XY,
)

x_train_c, y_train_c = datasets.dataset_to_tensors(
    poison_cifar_train.clean_dataset,
    None,  # batch_size * device_count,
    xmap=lambda x: np.moveaxis(x.numpy(), 0, -1),
)

x_train_p, y_train_p = datasets.dataset_to_tensors(
    poison_cifar_train.poison_dataset,
    None,  # batch_size * device_count,
    xmap=lambda x: np.moveaxis(x.numpy(), 0, -1),
)

x_train_pp, y_train_pp = datasets.dataset_to_tensors(
    poison_cifar_train.pre_poison_dataset,
    None,  # batch_size * device_count,
    xmap=lambda x: np.moveaxis(x.numpy(), 0, -1),
)

x_test_c, y_test_c = datasets.dataset_to_tensors(
    poison_cifar_test.clean_dataset,
    None,  # batch_size * device_count,
    xmap=lambda x: np.moveaxis(x.numpy(), 0, -1),
)

x_test_p, y_test_p = datasets.dataset_to_tensors(
    poison_cifar_test.poison_dataset,
    None,  # batch_size * device_count,
    xmap=lambda x: np.moveaxis(x.numpy(), 0, -1),
)

x_test_pp, y_test_pp = datasets.dataset_to_tensors(
    poison_cifar_test.pre_poison_dataset,
    None,  # batch_size * device_count,
    xmap=lambda x: np.moveaxis(x.numpy(), 0, -1),
)


def build_model(kernel_fn, g_dd, x_tr, y_tr, lam=1e-3):
    classes = y_tr.shape[1]
    predictor = nt.predict.gradient_descent_mse(
        g_dd, y_tr[:, np.newaxis, 0] - 1 / classes, diag_reg=lam
    )

    def model(
        x_te,
        t=None,
        batch_size=batch_size,
        device_count=device_count,
        store_on_device=True,
    ):
        ker = kernel_fn
        if batch_size is not None:
            ker = nt.batch(kernel_fn, batch_size, device_count, store_on_device)
        g_td = ker(x_te, x_tr, "ntk")
        return predictor(t, None, -1, g_td), g_td

    return model, predictor


def kernel_fit(
    kernel_fn,
    x_tr,
    y_tr,
    lam=0,
    batch_size=batch_size,
    device_count=-1,
    store_on_device=True,
):
    classes = y_tr.shape[1]
    g_dd = nt.batch(kernel_fn, batch_size, device_count, store_on_device)(
        x_tr, None, "ntk"
    )
    predictor = nt.predict.gradient_descent_mse(
        g_dd, y_tr[:, np.newaxis, 0] - 1 / classes, diag_reg=lam
    )

    def model(
        x_te,
        t=None,
        batch_size=batch_size,
        device_count=device_count,
        store_on_device=store_on_device,
    ):
        ker = kernel_fn
        if batch_size is not None:
            ker = nt.batch(kernel_fn, batch_size, device_count, store_on_device)
        g_td = ker(x_te, x_tr, "ntk")
        return predictor(t, None, -1, g_td), g_td

    return model, g_dd, predictor


# model, g_dd, predictor = kernel_fit(kernel_fn, x_train, y_train)

# np.save("cifar_01_gdd_1xs500.npy", g_dd)

# y_pred, g_td = model(x_test)
# acc = np.mean(y_pred.argmax(1) == y_test.argmax(1))

# np.save("cifar_01_gtd.npy", g_td)

# print(f"test accuracy: {acc}")

X = np.vstack([x_train_c, x_train_p, x_train_pp, x_test_c, x_test_p, x_test_pp])
