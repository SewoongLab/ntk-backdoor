import gc
import re
import io
import os
import copyreg
import warnings
import jax
import jaxlib
import pickle
import dill
import numpy as onp

from jax import random
from jax import numpy as np
from jax import scipy as sp
from jax.flatten_util import ravel_pytree
from jax.tree_util import (
    tree_flatten,
    tree_unflatten,
    tree_map,
    tree_reduce,
    tree_structure,
    tree_leaves,
)

import numpy as onp
import scipy as osp
import optax
from IPython import display
import pandas as pd
from collections import Counter
from itertools import count
import socket
from shapecheck import check_shapes

def natural_keys(text):
    """Key function for natural sort of strings."""
    atoi = lambda text: int(text) if text.isdigit() else text
    return [atoi(c) for c in re.split(r"(\d+)", text)]


backend = jax.lib.xla_bridge.get_backend()


def print_mem():
    for k, v in Counter(
        [(b.device().id, b.dtype.name, *b.shape) for b in backend.live_buffers()]
    ).items():
        print(f"{k}: {v}")


@check_shapes("D,D", "D,T", "D")
def predict(Kdd, Kdt, Y):
    return Y.T @ sp.linalg.solve(Kdd, Kdt, assume_a='pos')


@check_shapes("D,D", "D,T", "D")
def predict2(Kdd, Kdt, Y):
    return sp.linalg.solve(Kdd, Y, assume_a='pos') @ Kdt


@check_shapes("D,D", "D")
def predict_loocv(K, Y):
    Id = np.eye(len(K))
    Kc = K * (1 - Id)
    F = sp.linalg.cho_factor(K)
    Ki = sp.linalg.cho_solve(F, Id)
    KiY = sp.linalg.cho_solve(F, Y)
    KiKc = sp.linalg.cho_solve(F, Kc)
    Kii, Kiii = np.diag(K), np.diag(Ki)
    C = np.array([[Kii * Kiii, Kii * (1 - Kii * Kiii)], [-Kiii, Kii * Kiii]])
    KiU = np.dstack([Ki, KiKc])

    def do_1fold(kiu, c, kc):
        return kc @ KiY + (kc @ kiu @ np.linalg.inv(c)) @ (kiu[:, ::-1].T @ Y)

    return jax.vmap(do_1fold, (1, -1, 1))(KiU, C, Kc)


def bit_slice(B, s):
    return np.zeros_like(B, bool).at[B.nonzero()[0][s]].set(True)


cpu = jax.devices("cpu")[0]
gpu_kind = jax.devices()[0].device_kind
key = random.PRNGKey(0)


# from numpyro
def enable_x64(use_x64=True):
    """
    Changes the default array type to use 64 bit precision as in NumPy.

    :param bool use_x64: when `True`, JAX arrays will use 64 bits by default;
        else 32 bits.
    """
    if not use_x64:
        use_x64 = os.getenv("JAX_ENABLE_X64", 0)
    jax.config.update("jax_enable_x64", use_x64)


def disable_x64(use_x64=False):
    """
    Changes the default array type to use 32 bit precision.

    :param bool use_x64: when `True`, JAX arrays will use 64 bits by default;
        else 32 bits.
    """
    jax.config.update("jax_enable_x64", use_x64)


def set_platform(platform=None):
    """
    Changes platform to CPU, GPU, or TPU. This utility only takes
    effect at the beginning of your program.

    :param str platform: either 'cpu', 'gpu', or 'tpu'.
    """
    if platform is None:
        platform = os.getenv("JAX_PLATFORM_NAME", "cpu")
    jax.config.update("jax_platform_name", platform)


def set_host_device_count(n):
    """
    By default, XLA considers all CPU cores as one device. This utility tells XLA
    that there are `n` host (CPU) devices available to use. As a consequence, this
    allows parallel mapping in JAX :func:`jax.pmap` to work in CPU platform.

    .. note:: This utility only takes effect at the beginning of your program.
        Under the hood, this sets the environment variable
        `XLA_FLAGS=--xla_force_host_platform_device_count=[num_devices]`, where
        `[num_device]` is the desired number of CPU devices `n`.

    .. warning:: Our understanding of the side effects of using the
        `xla_force_host_platform_device_count` flag in XLA is incomplete. If you
        observe some strange phenomenon when using this utility, please let us
        know through our issue or forum page. More information is available in this
        `JAX issue <https://github.com/google/jax/issues/1408>`_.

    :param int n: number of CPU devices to use.
    """
    xla_flags = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(
        r"--xla_force_host_platform_device_count=\S+", "", xla_flags
    ).split()
    os.environ["XLA_FLAGS"] = " ".join(
        ["--xla_force_host_platform_device_count={}".format(n)] + xla_flags
    )


def block_until_ready(pytree):
    return tree_map(lambda t: t.block_until_ready(), pytree)


def device_put(pytree, device):
    return tree_map(lambda t: jax.device_put(t, device), pytree)


def sp_minimize(f, x0, *, disable_unary=False, **kwargs):
    unary = False
    if tree_structure(x0).num_nodes == 1 and not disable_unary:
        unary = True
        x0 = (x0,)

    # if isinstance(jax.eval_shape(f, *x0), jax.ShapeDtypeStruct):
    #     f = jax.jit(jax.value_and_grad(f))

    ty = np.float64 if jax.config.jax_enable_x64 else np.float32
    _, unravel = ravel_pytree(x0)

    def to_np(x):
        return tree_map(lambda t: t.astype(ty), unravel(x))

    def to_onp(x):
        return onp.asarray(ravel_pytree(x)[0]).astype(onp.float64)

    def f_wrapper(x):
        l, g = f(*to_np(x))
        return onp.array(l).astype(onp.float64), to_onp(g)

    inner_kwargs = {"jac": True, "method": "L-BFGS-B"}
    inner_kwargs.update(kwargs)

    if "callback" in inner_kwargs:
        callback = inner_kwargs["callback"]

        def callback_wrapper(xk, *args):
            return callback(*to_np(xk), *args)

        inner_kwargs["callback"] = callback_wrapper

    if "bounds" in inner_kwargs:
        bounds = inner_kwargs["bounds"]
        keep_feasible = False
        if isinstance(bounds, osp.optimize.Bounds):
            keep_feasible = bounds.keep_feasible
            bounds = (bounds.lb, bounds.ub)

        if isinstance(bounds, tuple):
            assert len(bounds) == 2
            lb, ub = bounds
            x0_shape = tree_map(lambda t: getattr(t, "shape", None), x0)
            lb_shape = tree_map(lambda t: getattr(t, "shape", None), lb)
            ub_shape = tree_map(lambda t: getattr(t, "shape", None), ub)
            if unary:
                assert lb_shape == ub_shape == x0_shape[0]
            else:
                assert lb_shape == ub_shape == x0_shape
            inner_kwargs["bounds"] = osp.optimize.Bounds(
                to_onp(lb), to_onp(ub), keep_feasible
            )

        else:
            raise NotImplementedError("Can only handle tuple and Bounds for bounds")

    if inner_kwargs["jac"] is not True:
        raise NotImplementedError("Only supports jac=True right now")

    # for p in ("hess", "hessp", "constraints"):
    #     if p in inner_kwargs:
    #         raise NotImplementedError(
    #             f"Have not implemented translation of {p} argument"
    #         )

    opt_min = osp.optimize.minimize(f_wrapper, to_onp(x0), **inner_kwargs)

    # TODO: This is super gross and could break things
    opt_min.fun = np.asarray(opt_min.fun)
    opt_min.x = to_np(opt_min.x)
    if unary:
        opt_min.x = opt_min.x[0]

    if hasattr(opt_min, "hess_inv"):
        hess_inv = opt_min.hess_inv

        def hess_inv_wrapper(x):
            # ravel takes care of unary input
            result = to_np(hess_inv(to_onp(x)))
            if unary:
                return result[0]
            return result

        opt_min.hess_inv = hess_inv_wrapper

    if hasattr(opt_min, "jac"):
        opt_min.jac = to_np(opt_min.jac)
        if unary:
            opt_min.jac = opt_min.jac[0]

    return opt_min


def optax_box_constraint_tx(lb, ub):
    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError

        updates = jax.tree_map(
            lambda p, u, a, b: np.where(p + u > b, b - p, np.where(p + u < a, a - p, u)),
            params, updates, lb, ub
        )
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)

def optax_minimize(f, x0, tx, *, bounds=None, disable_unary=False, verbose=True, callback=None, maxit=None):
    x = x0

    if bounds:
        keep_feasible = False
        if isinstance(bounds, osp.optimize.Bounds):
            bounds = (bounds.lb, bounds.ub)

        if isinstance(bounds, tuple):
            assert len(bounds) == 2
            lb, ub = bounds
            x0_shape = tree_map(lambda t: getattr(t, "shape", None), x0)
            lb_shape = tree_map(lambda t: getattr(t, "shape", None), lb)
            ub_shape = tree_map(lambda t: getattr(t, "shape", None), ub)
            assert lb_shape == ub_shape == x0_shape

            tx = optax.chain(tx, optax_box_constraint_tx(lb, ub))

        else:
            raise NotImplementedError("Can only handle tuple and Bounds for bounds")

    opt_state = tx.init(x)
    best_l = np.inf

    for epoch in range(maxit) if maxit else count(0):
        l, g = f(x)
        if l < best_l:
            best_l = l
            if callback is not None:
                callback(x, l)
        if verbose:
            print(f"{epoch} {l:.06E} best {best_l:.06E}")
        updates, opt_state = tx.update(g, opt_state, x)
        x = optax.apply_updates(x, updates)

    return osp.optimize.OptimizeResult(
        x=x,
        success=True,
        fun=l,
        jac=g,
        nfev=epoch+1,
        njev=epoch+1,
        nhev=0,
        nit=epoch+1,
    )



def vsplit(a, *block_sizes, check=True):
    blocks = []
    i = 0
    for bs in block_sizes:
        if not isinstance(bs, int):
            bs = len(bs)
        blocks.append(a[i : i + bs])
        i += bs
    if check:
        assert len(a) == i
    return blocks


def hsplit(a, *block_sizes, check=True):
    return [b.T for b in vsplit(a, *block_sizes, check=check)]


def learning_rate_schedule(n, init_lr, steps, step_lrs):
    def scheduler(i):
        epoch = i // n
        lr = init_lr
        for step, step_lr in zip(steps, step_lrs):
            lr = jax.lax.cond(
                epoch >= step, lambda _: step_lr, lambda _: lr, operand=None
            )
        return lr

    return scheduler


# https://github.com/iclr2022anon/fast_finite_width_ntk/blob/main/fast_finite_width_ntk/utils/utils.py#L458-L471
def get_flops(f, *a, optimize=True, **kw):
    m = jax.xla_computation(f)(*a, **kw)
    client = jax.lib.xla_bridge.get_backend()
    if optimize:
        m = client.compile(m).hlo_modules()[0]
    else:
        m = m.as_hlo_module()
    analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, m)

    if "flops" not in analysis:
        warnings.warn('No `"flops"` returned by HLO cost analysis.')
        return onp.inf

    return analysis["flops"]


GPU_MAP = {
    "Quadro RTX 6000": "rtx6k",
    "NVIDIA GeForce RTX 2080 Ti": "2080ti",
    "NVIDIA A40": "a40",
    "NVIDIA TITAN RTX": "rtx6k",
    "Tesla P100-PCIE-16GB": "rtx6k",
    "NVIDIA A100 80GB PCIe": "a100",
}


def platform_desc():
    devs = jax.devices()
    dev_kind = devs[0].device_kind
    dev_kind = GPU_MAP.get(dev_kind, dev_kind)
    dev_count = len(devs)
    bits = 64 if jax.config.jax_enable_x64 else 32
    host = socket.gethostname()
    return f"{host} with {dev_count}x {dev_kind} ({bits} bits)"


def platform_lookup(config):
    bits = 64 if jax.config.jax_enable_x64 else 32
    dev_kind = jax.devices()[0].device_kind
    return config[GPU_MAP.get(dev_kind, dev_kind), bits]


def binary_cross_entropy_with_logits(
    inputs: np.ndarray, targets: np.ndarray, average: bool = True
) -> np.ndarray:
    """Binary cross entropy loss.
    This function is based on the PyTorch implemantation.
    See : https://discuss.pytorch.org/t/numerical-stability-of-bcewithlogitsloss/8246
    Parameters
    ----------
    inputs : jnp.ndarray
        This is a model output. This is a value before passing a sigmoid function.
    targets : jnp.ndarray
        This is a label and the same shape as inputs.
    average : bool
        Whether to mean loss values or sum, default to be True.
    Returns
    -------
    loss : jnp.ndarray
        This is a binary cross entropy loss.
    """

    if inputs.shape != targets.shape:
        raise ValueError(
            "Target size ({}) must be the same as input size ({})".format(
                targets.shape, inputs.shape
            )
        )

    max_val = np.clip(-inputs, 0, None)
    loss = (
        inputs
        - inputs * targets
        + max_val
        + np.log(np.exp(-max_val) + np.exp((-inputs - max_val)))
    )

    if average:
        return np.mean(loss)

    return np.sum(loss)


class jax_pickle:
    @classmethod
    def dump(cls, obj, f):
        p = dill.Pickler(f)
        p.dispatch_table = copyreg.dispatch_table.copy()
        p.dispatch_table[jaxlib.xla_extension.DeviceArray] = lambda a: (
            np.asarray,
            (onp.asarray(a),),
        )
        p.dump(obj)

    @classmethod
    def dumps(cls, obj):
        f = io.BytesIO()
        cls.dump(obj, f)
        f.seek(0)
        return f.read()
