from pathlib import Path
from itertools import count
import pickle
import sys
import jax
import jax.numpy as np
from jax import jit, value_and_grad
from jax_utils import key, vsplit

import wandb
import optax
import models

run_name = "wrn34g-v9-r300"
use_wandb = True
save_checkpoints = 10
label_pair = (9, 4)
batch_size = 16

from neural_tangents import stax
init_fn, apply_fn, kernel_fn = models.WideResnet(block_size=4, k=5, num_classes=1)

root = Path(".")
checkpoint_dir = root / "output" / "checkpoints" / run_name
if not checkpoint_dir.exists():
    checkpoint_dir.mkdir()


def loss(params, batch):
    inputs, targets = batch
    outputs = apply_fn(params, inputs).ravel()
    return ((outputs - targets) ** 2).sum(), outputs


def update(i, opt_state, params, batch):
    inputs, targets = batch
    (l, outputs), g = value_and_grad(loss, has_aux=True)(params, batch)

    updates, opt_state = tx.update(g, opt_state, params)

    correct = np.sum(np.sign(outputs) == targets)
    return l, correct, opt_state, optax.apply_updates(params, updates)


def accuracy(opt_state, params, batch):
    inputs, targets = batch
    l, outputs = loss(params, batch)
    correct = np.sum(np.sign(outputs) == targets)
    return l, correct


lr = 1e-4 / batch_size

_, params = init_fn(key, input_shape=(1, 32, 32, 3))
tx = optax.sgd(lr, momentum=0.9)
opt_state = tx.init(params)

data_file = Path("output/X-94-1xs-perm.npy")
n, m = 5000, 1000
X = np.load(data_file)
x_train_c, x_train_p, x_train_pp, x_test_c, x_test_p, x_test_pp = vsplit(
    X, n, n, n, m, m, m
)
Xd = np.vstack([x_train_c, x_train_pp])
Yd = np.hstack([np.full(len(x_train_c), 1), np.full(len(x_train_pp), -1)])
# Xp = x_train_p
# Yp = np.full(len(x_train_p), 1)
Xt = np.vstack([x_test_c, x_test_pp])
Yt = np.hstack([np.full(len(x_test_c), 1), np.full(len(x_test_pp), -1)])
Xa = x_test_p
Ya = np.full(len(x_test_p), 1)

s, t = 8000, 0
D = np.s_[(2 * n - s) // 2 : -(2 * n - s) // 2 or None]
Xd, Yd = Xd[D], Yd[D]

# poison_file = "output/wrn34g-1xs/ebd-v9-10-tr2.2.npy"
# Xp = np.load(poison_file)
poison_file = None
# Xp = np.array([]).reshape(0, 32, 32, 3)
Xp = x_train_p[:300]
# mix = 32
# Xp = 1/mix * Xp + (1 - 1/mix) * Xd[s//2:][:10]
Yp = np.ones(len(Xp))
eps = len(Xp)

Xdp = np.vstack([Xd, Xp])
Ydp = np.hstack([Yd, Yp])

ttx = Xt.reshape(-1, 100, *Xt.shape[1:])
tty = Yt.reshape(-1, 100)

tax = Xa.reshape(-1, 100, *Xa.shape[1:])
tay = Ya.reshape(-1, 100)


@jit
def iter(epoch, params, opt_state, key):
    key, subkey = jax.random.split(key)

    P = jax.random.permutation(subkey, np.arange(len(Xdp)))
    tdx = Xdp[P][: len(Xdp) - len(Xdp) % batch_size].reshape(-1, batch_size, *Xdp.shape[1:])
    tdy = Ydp[P][: len(Xdp) - len(Xdp) % batch_size].reshape(-1, batch_size)

    def train_scanner(carry, xy):
        opt_state, params = carry
        l, correct, opt_state, params = update(epoch, opt_state, params, xy)
        return (opt_state, params), (l, correct)

    (opt_state, params), (l, correct) = jax.lax.scan(
        train_scanner, (opt_state, params), (tdx, tdy)
    )

    avg_loss, avg_acc = l.sum(), correct.sum()
    leftovers = (
        Xdp[P][-(len(Xdp) % batch_size) or len(Xdp):],
        Ydp[P][-(len(Xdp) % batch_size) or len(Xdp):]
    )
    l, correct, opt_state, params = update(epoch, opt_state, params, leftovers)
    avg_loss += l
    avg_acc += correct
    avg_loss /= len(Xdp)
    avg_acc /= len(Xdp)

    def acc_mapper(xy):
        return accuracy(opt_state, params, xy)

    l, correct = jax.lax.map(acc_mapper, (ttx, tty))
    avg_tloss = l.sum() / len(Xt)
    avg_tacc = correct.sum() / len(Xt)

    l, correct = jax.lax.map(acc_mapper, (tax, tay))
    avg_ploss = l.sum() / len(Xa)
    avg_pacc = correct.sum() / len(Xa)

    if eps > 0:
        p_pred = apply_fn(params, Xp)
    else:
        p_pred = np.zeros(1)

    return (params, opt_state, key), (
        avg_loss,
        avg_acc,
        avg_tloss,
        avg_tacc,
        avg_ploss,
        avg_pacc,
        p_pred,
    )


# if __name__ == "__main__":
#     if use_wandb:
#         wandb.init(
#             project="ntk-backdoor",
#             entity="jhayase",
#             settings=wandb.Settings(start_method="fork"),
#         )
#         wandb.run.name = run_name
#         print(f"Running {wandb.run.name}")
#         wandb.config.update(
#             {
#                 "learning_rate": lr,
#                 "batch_size": batch_size,
#                 "eps": eps,
#                 "dataset": "cifar10",
#                 "source_label": label_pair[0],
#                 "target_label": label_pair[1],
#                 "optimizer": "sgd",
#                 "poison_type": "grad_opt",
#                 "poison_trigger": "periodic",
#                 "poison_source": poison_file,
#                 "model": "wrn34g-5x",
#                 "float_bits": 64 if jax.config.FLAGS.jax_enable_x64 else 32,
#                 "save_checkpoints": save_checkpoints,
#             }
#         )
#     key = jax.random.PRNGKey(0)
#     for epoch in count(0):
#         (params, opt_state, key), (
#             avg_loss,
#             avg_acc,
#             avg_tloss,
#             avg_tacc,
#             avg_ploss,
#             avg_pacc,
#             p_pred,
#         ) = iter(epoch, params, opt_state, key)

#         print(
#             f"{run_name}: {epoch:04d}, "
#             f"tr: ({avg_loss:.07f}, {avg_acc:.04f}), "
#             f"te: ({avg_tloss:.05f}, {avg_tacc:.04f}), "
#             f"p: ({avg_ploss:.05f}, {avg_pacc:.03f}), "
#             f"{p_pred.mean():+.06f}"
#         )

#         if use_wandb:
#             wandb.log(
#                 {
#                     "train_loss": avg_loss,
#                     "train_acc": avg_acc,
#                     "test_loss": avg_tloss,
#                     "test_acc": avg_tacc,
#                     "poison_loss": avg_ploss,
#                     "poison_acc": avg_pacc,
#                     "poison_pred": float(p_pred.mean()),
#                 }
#             )

#         if save_checkpoints and epoch % save_checkpoints == 0:
#             pickle.dump(params, open(checkpoint_dir / f"params_{epoch:06d}.pkl", "wb"))
#             pickle.dump(
#                 {"epoch": epoch, "opt_state": opt_state},
#                 open(checkpoint_dir / "state.pkl", "wb"),
#             )

#         sys.stdout.flush()
#         if avg_loss < 1e-6:
#             break
