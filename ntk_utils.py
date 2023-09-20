import jax
from jax import vmap, grad, pmap, jvp, value_and_grad, jacrev
import jax.numpy as np
from jax.tree_util import tree_map, tree_reduce
from jax_utils import key
from functools import wraps, partial
import collections
import pickle
import operator as op
import neural_tangents as nt

# from fast_finite_width_ntk import empirical as fast_empirical
from pathlib import Path
from shapecheck import check_shapes


def scalarize(fn):
    @wraps(fn)
    def scalar_fn(*args, **kwargs):
        result = fn(*args, **kwargs)
        assert result.size == 1
        return result.ravel()[0]

    return scalar_fn


@check_shapes(x="N,H,W,C", y="M,H,W,C", w="N,M")
def kgrad_td_device(kf, wrap, x, y, w, argnum, batch_size):
    n = len(y)
    n_even = n - n % batch_size
    y_batches = y[:n_even].reshape((-1, batch_size, *y.shape[1:]))
    wT_batches = w.T[:n_even].reshape((-1, batch_size, len(w)))

    if wrap:

        def vg_kf(xs, ys):
            return vmap(grad(scalarize(kf), argnums=argnum), (None, 0))(
                xs, ys[:, np.newaxis]
            )

    else:

        def vg_kf(xs, ys):
            # I do find the need for moveaxis here a little unsettling
            return np.moveaxis(kf(xs, ys), 1, 0)

    # TODO: Maybe use a better summation alg to reduce floating point error
    def kgrad_td_device_scanner(s, yw):
        (y_batch, wT_batch) = yw

        def kgrad_td_row_scanner(s, xw):
            (x_single, w_single) = xw
            print(s.shape, x_single.shape, y_batch.shape, w_single.shape)
            return (
                s
                + np.tensordot(
                    w_single,
                    vg_kf(x_single, y_batch),
                    axes=1,
                )
            ), None

        return (
            jax.lax.scan(
                kgrad_td_row_scanner,
                s,
                (x[:, np.newaxis], wT_batch.T),
            )[0],
            None,
        )

    v = jax.lax.scan(
        kgrad_td_device_scanner,
        np.zeros((1, *y.shape[1:])),
        (y_batches, wT_batches),
    )[0]

    if n % batch_size != 0:
        y_batch, wT_batch = y[n_even:], w.T[n_even:]
        v = kgrad_td_device_scanner(v, (y_batch, wT_batch))[0]

    return v


@check_shapes(x="N,H,W,C", y="M,H,W,C", w="N,M")
def kgrad_td(kf, x, y, w, argnum=0, batch_size=5, device_count=None, wrap=True):
    # assert w.shape == (x.shape[0], y.shape[0])
    # assert x.shape[1:] == y.shape[1:]

    dev_count = device_count or len(jax.devices())
    n = len(y)

    pad = -n % dev_count
    y_padded = np.vstack([y, np.ones((pad, *y.shape[1:]))])
    y_batches = y_padded.reshape((dev_count, -1, *y.shape[1:]))
    w_padded = np.hstack([w, np.zeros((len(w), pad))])
    w_batches = w_padded.reshape((len(w_padded), dev_count, -1))

    return pmap(
        kgrad_td_device,
        in_axes=(None, None, None, 0, 1, None, None),
        static_broadcasted_argnums=(0, 1, 5, 6),
        devices=jax.devices()[-dev_count:],
    )(kf, wrap, x, y_batches, w_batches, argnum, batch_size).sum(axis=0)


@check_shapes(x="N,H,W,C", y="M,H,W,C", w="N,M")
def kgrad_td_rows(kf, x, y, w, argnum=0, batch_size=5, device_count=None, wrap=True):
    return np.vstack(
        kgrad_td(
            kf,
            xp[np.newaxis],
            y,
            wp[np.newaxis],
            argnum=argnum,
            batch_size=batch_size,
            device_count=device_count,
            wrap=wrap,
        )
        for xp, wp in zip(x, w)
    )


@check_shapes(x="1,H,W,C", y="M,H,W,C", w="1,M")
def kgrad_td_fmap_device(apply_fn, params, x, y, w, batch_size):
    n = len(y)
    n_even = n - n % batch_size
    y_batches = y[:n_even].reshape((-1, batch_size, *y.shape[1:]))
    wT_batches = w.T[:n_even].reshape((-1, batch_size, len(w)))

    def fmap(zs):
        return vmap(partial(grad(scalarize(apply_fn)), params))(zs[:, None])

    def scanner(tangent, yw):
        (y_batch, wT_batch) = yw
        print(y_batch.shape, wT_batch.shape)
        yf_batch = fmap(y_batch)
        return (
            tree_map(
                lambda t, yf: t + np.tensordot(wT_batch.T, yf, 1)[0], tangent, yf_batch
            ),
            None,
        )

    tangent = tree_map(np.zeros_like, params)
    tangent = jax.lax.scan(scanner, tangent, (y_batches, wT_batches))[0]

    if n % batch_size != 0:
        y_batch, wT_batch = y[n_even:], w.T[n_even:]
        tangent = scanner(tangent, (y_batch, wT_batch))[0]

    dx = grad(
        lambda x: jax.jvp(scalarize(partial(apply_fn, inputs=x)), [params], [tangent])[
            1
        ]
    )(x)

    return dx


@check_shapes(x="1,H,W,C", y="M,H,W,C", w="1,M")
def kgrad_td_fmap(apply_fn, params, x, y, w, batch_size=5, device_count=None):
    dev_count = device_count or len(jax.devices())
    n = len(y)

    pad = -n % dev_count
    y_padded = np.vstack([y, np.ones((pad, *y.shape[1:]))])
    y_batches = y_padded.reshape((dev_count, -1, *y.shape[1:]))
    w_padded = np.hstack([w, np.zeros((len(w), pad))])
    w_batches = w_padded.reshape((len(w_padded), dev_count, -1))

    return pmap(
        kgrad_td_fmap_device,
        in_axes=(None, None, None, 0, 1, None, None),
        static_broadcasted_argnums=(0, 5),
        devices=jax.devices()[-dev_count:],
    )(apply_fn, params, x, y_batches, w_batches, batch_size).sum(axis=0)


@check_shapes(x="N,H,W,C", y="M,H,W,C", w="N,M")
def kgrad_td_rows_fmap(apply_fn, params, x, y, w, batch_size=5, device_count=None):
    return np.vstack(
        kgrad_td_fmap(
            apply_fn,
            params,
            xp[np.newaxis],
            y,
            wp[np.newaxis],
            batch_size=batch_size,
            device_count=device_count,
        )
        for xp, wp in zip(x, w)
    )


@check_shapes(xs="N,H,W,C", ys="M,H,W,C", ws="N,M")
def kgrad_td_rows_fmap2_device(apply_fn, params, xs, ys, ws, batch_size):
    n = len(ys)
    n_even = n - n % batch_size
    y_batches = ys[:n_even].reshape((-1, batch_size, *ys.shape[1:]))
    wT_batches = ws.T[:n_even].reshape((-1, batch_size, len(ws)))

    def fmap(zs):
        return vmap(partial(grad(scalarize(apply_fn)), params))(zs[:, None])

    def scanner(tangent, yw):
        (y_batch, wT_batch) = yw
        print(y_batch.shape, wT_batch.shape)
        yf_batch = fmap(y_batch)
        return (
            tree_map(
                lambda t, yf: t + np.tensordot(wT_batch.T, yf, 1), tangent, yf_batch
            ),
            None,
        )

    tangents = tree_map(lambda p: np.zeros((len(xs), *p.shape)), params)
    tangents = jax.lax.scan(scanner, tangents, (y_batches, wT_batches))[0]

    if n % batch_size != 0:
        y_batch, wT_batch = ys[n_even:], ws.T[n_even:]
        tangents = scanner(tangents, (y_batch, wT_batch))[0]

    def pullback(x, tangent):
        return grad(
            lambda x: jax.jvp(
                scalarize(partial(apply_fn, inputs=x)), [params], [tangent]
            )[1]
        )(x[None])

    dxs = vmap(pullback, (0, 0))(xs, tangents)

    return dxs[:, 0]


@check_shapes(x="N,H,W,C", y="M,H,W,C", w="N,M")
def kgrad_td_rows_fmap2(apply_fn, params, x, y, w, batch_size=5, device_count=None):
    dev_count = device_count or len(jax.devices())
    n = len(y)

    pad = -n % dev_count
    y_padded = np.vstack([y, np.ones((pad, *y.shape[1:]))])
    y_batches = y_padded.reshape((dev_count, -1, *y.shape[1:]))
    w_padded = np.hstack([w, np.zeros((len(w), pad))])
    w_batches = w_padded.reshape((len(w_padded), dev_count, -1))

    return pmap(
        kgrad_td_rows_fmap2_device,
        in_axes=(None, None, None, 0, 1, None, None),
        static_broadcasted_argnums=(0, 5),
        devices=jax.devices()[-dev_count:],
    )(apply_fn, params, x, y_batches, w_batches, batch_size).sum(axis=0)


@check_shapes(xs="N,H,W,C", ys="M,H,W,C")
def keval_device(kf, xs, ys, batch_size):
    n = len(ys)
    n_even = n - n % batch_size
    y_batches = ys[:n_even].reshape(-1, batch_size, *xs.shape[1:])

    def keval_scanner(_, y_batch):
        def keval_row_scanner(_, x_single):
            # print(f"{x_single.shape}, {y_batch.shape}")
            return None, kf(x_single, y_batch)

        return None, jax.lax.scan(keval_row_scanner, None, xs[:, np.newaxis])[1]

    v = jax.lax.scan(keval_scanner, None, y_batches)[1]
    v = np.moveaxis(v, 0, 1).reshape(xs.shape[0], n_even)

    if n % batch_size != 0:
        y_batch = ys[n_even:]
        # TODO: This could be done faster by transposing The
        # leftovers, using the same scanner on that, then doing a
        # final cleanup on the remaining corner.
        v = np.hstack([v, keval_scanner(None, y_batch)[1][:, 0]])

    return v


@check_shapes(xs="N,H,W,C", ys="M,H,W,C")
def keval_device2(kf, xs, ys, batch_size):
    n = len(ys)
    n_even = n - n % batch_size
    y_batches = ys[:n_even].reshape(-1, batch_size, *xs.shape[1:])

    def keval_mapper(y_batch):
        def keval_row_mapper(x_single):
            print(f"{x_single.shape}, {y_batch.shape}")
            return kf(x_single, y_batch)

        return jax.lax.map(keval_row_mapper, xs[:, np.newaxis])

    v = jax.lax.map(keval_mapper, y_batches)
    v = np.moveaxis(v, 0, 1).reshape(xs.shape[0], n_even)

    if n % batch_size != 0:
        y_batch = ys[n_even:]
        # TODO: This could be done faster by transposing The
        # leftovers, using the same scanner on that, then doing a
        # final cleanup on the remaining corner.
        v = np.hstack([v, keval_mapper(y_batch)[:, 0]])

    return v


@check_shapes(xs="N,H,W,C", ys="M,H,W,C")
def keval(kf, xs, ys, batch_size=5, device_count=None):
    dev_count = device_count or len(jax.devices())
    n = len(ys)
    pad = -n % dev_count
    y_padded = np.vstack([ys, np.zeros((pad, *ys.shape[1:]))])
    y_chunks = y_padded.reshape(dev_count, -1, *ys.shape[1:])
    v = pmap(
        keval_device,
        in_axes=(None, None, 0, None),
        static_broadcasted_argnums=(0, 3),
        devices=jax.devices()[:dev_count],
    )(kf, xs, y_chunks, batch_size)
    return np.moveaxis(v, 0, 1).reshape(xs.shape[0], ys.shape[0] + pad)[
        :, : -pad or None
    ]


@check_shapes(xs="N,H,W,C", ys="M,H,W,C")
def keval_fmap_fwd_device(apply_fn, params, xs, ys, batch_size):
    n = len(ys)
    n_even = n - n % batch_size
    y_batches = ys[:n_even].reshape(-1, batch_size, *xs.shape[1:])
    xf = vmap(partial(grad(scalarize(apply_fn)), params))(xs[:, None])

    def keval_mapper(y_batch):
        def keval_row_mapper(xf_single):
            return vmap(
                lambda inputs: jvp(
                    partial(apply_fn, inputs=inputs), [params], [xf_single]
                )
            )(y_batch[:, None])[1][..., 0, 0]

        return jax.lax.map(keval_row_mapper, xf)

    v = jax.lax.map(keval_mapper, y_batches)
    v = np.moveaxis(v, 0, 1).reshape(xs.shape[0], n_even)

    if n % batch_size != 0:
        y_batch = ys[n_even:]
        v = np.hstack([v, keval_mapper(y_batch)])

    return v


@check_shapes(xs="N,H,W,C", ys="M,H,W,C")
def keval_fmap_rev_device(apply_fn, params, xs, ys, batch_size):
    n = len(ys)
    n_even = n - n % batch_size
    y_batches = ys[:n_even].reshape(-1, batch_size, *xs.shape[1:])

    def fmap(zs):
        return vmap(partial(grad(scalarize(apply_fn)), params))(zs[:, None])

    xf = fmap(xs)

    def keval_mapper(y_batch):
        yf = fmap(y_batch)
        return tree_reduce(
            op.add,
            tree_map(
                lambda x, y: jax.lax.dot_general(
                    x, y, ([range(1, len(x.shape))] * 2, ((), ()))
                ),
                xf,
                yf,
            ),
        )

    v = jax.lax.map(keval_mapper, y_batches)
    v = np.moveaxis(v, 0, 1).reshape(xs.shape[0], n_even)

    if n % batch_size != 0:
        y_batch = ys[n_even:]
        v = np.hstack([v, keval_mapper(y_batch)])

    return v


@check_shapes(xs="N,H,W,C", ys="M,H,W,C")
def keval_fmap(apply_fn, params, xs, ys, batch_size=5, device_count=None, mode="rev"):
    assert mode in {"fwd", "rev"}

    dev_count = device_count or len(jax.devices())
    n = len(ys)
    pad = -n % dev_count
    y_padded = np.vstack([ys, np.zeros((pad, *ys.shape[1:]))])
    y_chunks = y_padded.reshape(dev_count, -1, *ys.shape[1:])
    v = pmap(
        keval_fmap_fwd_device if mode == "fwd" else keval_fmap_rev_device,
        in_axes=(None, None, None, 0, None),
        static_broadcasted_argnums=(0, 4),
        devices=jax.devices()[:dev_count],
    )(apply_fn, params, xs, y_chunks, batch_size)
    return np.moveaxis(v, 0, 1).reshape(xs.shape[0], ys.shape[0] + pad)[
        :, : -pad or None
    ]


def custom_empirical_ntk_fn(apply_fn, params, implementation=3):
    if implementation in (1, 2, 3):
        ekf = nt.empirical_ntk_fn(apply_fn, vmap_axes=0, implementation=implementation)
    # elif implementation == 3:
    #     ekf = fast_empirical.empirical_ntk_fn(
    #         apply_fn, vmap_axes=0, implementation=implementation
    #     )

    kernel_fn = partial(ekf, params=params)
    return kernel_fn


def load_empirical_kernel(
    model, params=None, key=key, input_shape=(1, 32, 32, 3), implementation=3
):
    if params is None:
        # get the params from init_fn
        assert len(model) == 3
        init_fn = model[0]
        _, params = init_fn(key, input_shape)
    elif isinstance(params, (str, Path)):
        params = Path(params)
        # load the params from a file
        if params.suffix == ".npy":
            params = np.load(params)
        else:
            params = pickle.load(open(params, "rb"))
    elif isinstance(params, np.array):
        pass
    else:
        raise NotImplementedError()

    params = tree_map(lambda t: np.asarray(t).astype(np.float64), params)

    if not isinstance(model, collections.abc.Callable):
        apply_fn = model[1]
    else:
        apply_fn = model

    kernel_fn = custom_empirical_ntk_fn(apply_fn, params, implementation)

    return kernel_fn, apply_fn, params


def make_empirical_kernel_grad(apply_fn, params, argnum=0, implementation=1):
    if implementation in (1, 2, 3):
        kernel_fn = custom_empirical_ntk_fn(apply_fn, params, implementation)
        ekg_core = value_and_grad(scalarize(kernel_fn), argnum)

    elif implementation == 4:
        if argnum == 0:

            def ekg_core(x, y):
                return jvp(
                    lambda p: value_and_grad(scalarize(apply_fn), 1)(p, x),
                    (params,),
                    (jacrev(scalarize(apply_fn))(params, y),),
                )[1]

        elif argnum == 1:

            def ekg_core(x, y):
                return jvp(
                    lambda p: value_and_grad(scalarize(apply_fn), 1)(p, y),
                    (params,),
                    (jacrev(scalarize(apply_fn))(params, x),),
                )[1]

        else:
            raise NotImplementedError

    def ekg(xs, ys):
        result = vmap(
            vmap(ekg_core, (None, 0)),
            (0, None),
        )(xs[:, np.newaxis], ys[:, np.newaxis])

        return result[0], result[1][:, :, 0]

    return ekg
