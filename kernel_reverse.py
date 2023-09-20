import jax
import jax.numpy as np
import jax.scipy as sp
import numpy as onp
import ntk_utils
import neural_tangents as nt

from jax import pmap, vmap, jit, grad, value_and_grad
from ntk_utils import keval, kgrad_td_rows, keval_fmap, kgrad_td_rows_fmap2
from jax_utils import (
    predict,
    predict2,
    bit_slice,
    cpu,
    gpu_kind,
    key,
    sp_minimize,
    vsplit,
    hsplit,
    device_put,
    platform_desc,
    platform_lookup,
    enable_x64,
    disable_x64,
)
from util import str_slice
from neural_tangents import stax

import models
import datasets

from pathlib import Path
from functools import partial
from shapecheck import check_shapes

enable_x64()

# codes
# d: clean train data
# p: poisoned train data
# t: clean test data
# a: poisoned (attack) test data

K_BATCH_SIZE, KG_BATCH_SIZE = platform_lookup(
    {
        ("cpu", 32): (20, 20),
        ("cpu", 64): (20, 20),
        ("2080ti", 32): (10, 10),
        ("2080ti", 64): (5, 1),
        ("rtx6k", 32): (40, 20),
        ("rtx6k", 64): (20, 10),
        ("a40", 32): (80, 80),
        ("a40", 64): (40, 40),
        ("a100", 32): (160, 160),
        ("a100", 64): (80, 80),
    }
)


# K_BATCH_SIZE, KG_BATCH_SIZE = platform_lookup(
#     {
#         ("cpu", 32): (20, 20),
#         ("cpu", 64): (20, 20),
#         ("2080ti", 32): (10, 10),
#         ("2080ti", 64): (5, 1),
#         ("rtx6k", 32): (40, 20),
#         ("rtx6k", 64): (20, 10),
#         ("a40", 32): (80, 80),
#         ("a40", 64): (30, 40),
#         ("a100", 32): (160, 160),
#         ("a100", 64): (60, 60),
#     }
# )


@check_shapes("D,D", "P,D", "P,P")
def bd_build_kdd(Kdd, Kpd, Kpp):
    return np.block([[Kdd, Kpd.T], [Kpd, Kpp]])


@check_shapes("D,T", "P,T")
def bd_build_kdt(Kdt, Kpt):
    return np.vstack([Kdt, Kpt])


def bd_build_y(Yd, Yp):
    return np.hstack([Yd, Yp])


@check_shapes("D,D", "P,D", "P,P", "D,T", "P,T", "D", "P")
def bd_predict(Kdd, Kpd, Kpp, Kdt, Kpt, Yd, Yp):
    return predict2(
        bd_build_kdd(Kdd, Kpd, Kpp), bd_build_kdt(Kdt, Kpt), bd_build_y(Yd, Yp)
    )


# @jit
@check_shapes("D,D", "D,T", "D", "P,D", "P,T", "P", "P")
def predict_add_ps(Kdd, Kdt, Yd, Kpd, Kpt, Kpp_diag, Yp):
    eps = np.sqrt(np.finfo(Kdd.dtype).eps)
    F = sp.linalg.cho_factor(Kdd)
    beta = sp.linalg.cho_solve(F, Yd)
    es = sp.linalg.cho_solve(F, Kpd.T)
    y_pred = Kdt.T @ beta
    Ss = 1 / (Kpp_diag - (Kpd * es.T).sum(1) + eps)
    beta_ps = np.vstack(
        [
            beta[:, None] + Ss * (Yd.T @ es) * es - es * Ss * Yp,
            -(es * Yd[:, None]).sum(0) * Ss + Yp * Ss,
        ]
    )
    y_pred_ps = Kdt.T @ beta_ps[:-1] + beta_ps[-1] * Kpt.T
    return y_pred, y_pred_ps.T


# @jit
@check_shapes("D,D", "D,T", "D", "P,D", "P,T", "P", "P")
def predict_add_ps2(Kdd, Kdt, Yd, Kpd, Kpt, Kpp_diag, Yp):
    eps = np.sqrt(np.finfo(Kdd.dtype).eps)
    F = sp.linalg.cho_factor(Kdd)
    beta = sp.linalg.cho_solve(F, Yd)
    es = sp.linalg.cho_solve(F, Kpd.T)
    y_pred = beta @ Kdt
    Ss = 1 / (Kpp_diag - (Kpd * es.T).sum(1) + eps)
    beta_ps = np.vstack(
        [
            beta[:, None] + Ss * (Yd.T @ es) * es - es * Ss * Yp,
            -(es * Yd[:, None]).sum(0) * Ss + Yp * Ss,
        ]
    )
    # print(beta_ps[:-1].shape, Kdt.shape, beta_ps[-1].shape, Kpt.shape)
    y_pred_ps = beta_ps[:-1].T @ Kdt + beta_ps[-1, None].T * Kpt
    return y_pred, y_pred_ps


@check_shapes("D,D", "D,T", "D", "P,D", "P,T", "P,P", "P")
def bd_greedy(Kdd, Kdt, Yd, Kpd, Kpt, Kpp, Yp, eps=10, verbose=True):
    S, Sc = np.arange(len(Yp)), np.empty(0, dtype=int)

    for round in range(eps):
        y_pred, y_pred_ps = predict_add_ps2(
            np.block([[Kdd, Kpd[Sc].T], [Kpd[Sc], Kpp[Sc][:, Sc]]]),
            np.vstack([Kdt, Kpt[Sc]]),
            np.hstack([Yd, Yp[Sc]]),
            np.hstack([Kpd[S], Kpp[S][:, Sc]]),
            Kpt[S],
            np.diag(Kpp)[S],
            Yp[S],
        )
        loss = ((y_pred - 1).clip(None, 0) ** 2).sum()
        losses = ((y_pred_ps - 1).clip(None, 0) ** 2).sum(1)
        best_j = losses.argmin()
        best_loss = losses[best_j]
        if verbose:
            print(f"selected {S[best_j]}: {loss:.2f} (diff {loss - best_loss:.2f})")
        Sc = np.append(Sc, S[best_j])
        S = np.delete(S, best_j)

    return Sc


@check_shapes("D,D", "D,T", "D", "P,D", "P,T", "P,P", "P")
def bd_greedy2(Kdd, Kdt, Yd, Kpd, Kpt, Kpp, Yp, eps=10, verbose=True):
    S, Sc = np.arange(len(Yp)), np.empty(0, dtype=int)

    for round in range(eps):
        y_pred, y_pred_ps = predict_add_ps2(
            np.block([[Kdd, Kpd[Sc].T], [Kpd[Sc], Kpp[Sc][:, Sc]]]),
            np.vstack([Kdt, Kpt[Sc]]),
            np.hstack([Yd, Yp[Sc]]),
            np.hstack([Kpd[S], Kpp[S][:, Sc]]),
            Kpt[S],
            np.diag(Kpp)[S],
            Yp[S],
        )
        loss = ((y_pred - 1).clip(None, 0) ** 2).sum()
        losses = ((y_pred_ps - 1).clip(None, 0) ** 2).sum(1)
        best_j = losses.argmin()
        best_loss = losses[best_j]
        if verbose:
            asr = np.mean(y_pred_ps[best_j] > 0)
            print(
                f"{round} selected {S[best_j]}: {loss:.2f} → {best_loss:.2f} (diff {loss - best_loss:.2f}) {asr*100:.2f}% ASR"
            )
        Sc = np.append(Sc, S[best_j])
        S = np.delete(S, best_j)

    round_loss = best_loss
    while True:
        print("=" * 50)
        for round in range(eps):
            S = np.append(S, Sc[round])
            Sc = np.delete(Sc, round)
            y_pred, y_pred_ps = predict_add_ps2(
                np.block([[Kdd, Kpd[Sc].T], [Kpd[Sc], Kpp[Sc][:, Sc]]]),
                np.vstack([Kdt, Kpt[Sc]]),
                np.hstack([Yd, Yp[Sc]]),
                np.hstack([Kpd[S], Kpp[S][:, Sc]]),
                Kpt[S],
                np.diag(Kpp)[S],
                Yp[S],
            )
            losses = np.nan_to_num(
                ((y_pred_ps - 1).clip(None, 0) ** 2).sum(1), nan=float("inf")
            )
            loss = losses[-1]
            best_j = losses.argmin()
            best_loss = losses[best_j]
            if verbose:
                asr = np.mean(y_pred_ps[best_j] > 0)
                print(
                    f"{round} selected {S[best_j]}: {loss:.2f} → {best_loss:.2f} (diff {loss - best_loss:.2f}) {asr*100:.2f}% ASR"
                )
            Sc = np.insert(Sc, round, S[best_j])
            S = np.delete(S, best_j)
        if best_loss < round_loss:
            round_loss = best_loss
        else:
            break

    return Sc


def bd_split_G(G, num_train=5000, num_test=1000):
    k, n, m = len(G), num_train, num_test
    d = np.full(k, False).at[:n].set(True).at[2 * n : 3 * n].set(True)
    p = np.full(k, False).at[n : 2 * n].set(True)
    t = (
        np.full(k, False)
        .at[3 * n : 3 * n + m]
        .set(True)
        .at[3 * n + 2 * m : 3 * n + 3 * m]
        .set(True)
    )
    a = np.full(k, False).at[3 * n + m : 3 * n + 2 * m].set(True)
    Kdd = G[d][:, d]
    Kpd = G[p][:, d]
    Kpp = G[p][:, p]

    Kdt = G[d][:, t]
    Kpt = G[p][:, t]
    Kda = G[d][:, a]
    Kpa = G[p][:, a]

    return Kdd, Kpd, Kpp, Kdt, Kpt, Kda, Kpa


def bd_make_Y(num_train=5000, num_test=1000):
    Yd = np.hstack([np.full(num_train, 1), np.full(num_train, -1)])
    Yp = np.full(num_train, 1)
    Yt = np.hstack([np.full(num_test, 1), np.full(num_test, -1)])
    Ya = np.full(num_test, 1)
    return Yd, Yp, Yt, Ya


@check_shapes("D,D", "P,D", "P,P", "D,T", "D,A", "P,T", "P,A", "D", "P", "T")
def bd_loss(Kdd, Kpd, Kpp, Kdt, Kda, Kpt, Kpa, Yd, Yp, Yt):
    # print(Kdd, Kpd, Kpp, Kdt, Kda, Kpt, Kpa, Yd, Yp, Yt)
    yt_pred, ya_pred = vsplit(
        bd_predict(Kdd, Kpd, Kpp, np.hstack([Kdt, Kda]), np.hstack([Kpt, Kpa]), Yd, Yp),
        Kpt.T,
        Kpa.T,
    )
    # return yt_pred, ya_pred
    # return Kdd, Kpd, Kpp, Kdt, Kda, Kpt, Kpa, Yd, Yp, Yt
    return np.sum((yt_pred - Yt) ** 2) + np.sum((ya_pred - 1).clip(None, 0) ** 2), (
        yt_pred,
        ya_pred,
    )


bd_loss_grad = jit(
    value_and_grad(bd_loss, argnums=(1, 2, 5, 6), has_aux=True), backend="cpu"
)


@check_shapes("D,D", "D")
def predict_k_fold(Kdd, Yd, k=5):
    assert len(Kdd) % 2 == 0
    n = len(Kdd) // 2
    assert n % k == 0
    m = n // k
    S = jax.lax.dynamic_slice

    def bd_loss_per_fold(fold):
        D = (
            ((np.arange(k) != fold).nonzero(size=k - 1)[0] * m)[np.newaxis].T
            + np.array([0, n])
        ).T.ravel()
        T = fold * m + np.array([0, n])
        return predict2(
            Kdd=np.block([[S(Kdd, (R, C), (m, m)) for C in D] for R in D]),
            Kdt=np.block([[S(Kdd, (R, C), (m, m)) for C in T] for R in D]),
            Y=np.hstack([S(Yd, (C,), (m,)) for C in D]),
        )

    result = jax.lax.map(bd_loss_per_fold, np.arange(k))
    return result


@check_shapes("D,D", "D", "P,D", "P,P", "P")
def bd_greedy_k_fold(Kdd, Yd, Kpd, Kpp, Yp, k=10, eps=10):
    Sp, Sc = np.arange(len(Yp)), np.empty(0, dtype=int)
    S = jax.lax.dynamic_slice
    n, p = len(Kdd) // 2, len(Kpp)
    assert n % k == 0
    m = n // k

    def predict_add_ps_per_fold(fold, Sp, Sc):
        D = (
            ((np.arange(k) != fold).nonzero(size=k - 1)[0] * m)[np.newaxis].T
            + np.array([0, n])
        ).T.ravel()
        # T = fold * m + np.array([0, n])
        Kdd2 = np.block([[S(Kdd, (R, C), (m, m)) for C in D] for R in D])
        # Kdt2 = np.block([[S(Kdd, (R, C), (m, m)) for C in T] for R in D])
        Kpd2 = np.block([[S(Kpd, (0, C), (p, m)) for C in D]])
        # Kpt2 = np.block([[S(Kpd, (0, C), (p, m)) for C in T]])
        Yd2 = np.hstack([S(Yd, (C,), (m,)) for C in D])
        y_pred, y_pred_ps = predict_add_ps(
            np.block([[Kdd2, Kpd2[Sc].T], [Kpd2[Sc], Kpp[Sc][:, Sc]]]),
            np.vstack([Kpd2.T, Kpp[Sc]]),
            np.hstack([Yd2, Yp[Sc]]),
            np.hstack([Kpd2[Sp], Kpp[Sp][:, Sc]]),
            Kpp[Sp],
            np.diag(Kpp)[Sp],
            Yp[Sp],
        )

        loss = ((y_pred - 1).clip(None, 0) ** 2).sum()
        losses = ((y_pred_ps - 1).clip(None, 0) ** 2).sum(1)
        return loss, losses

    # result = jax.lax.map(lambda fold: predict_add_ps_per_fold(fold, Sp, Sc), np.arange(k))
    result = predict_add_ps_per_fold(0, Sp, Sc)
    return result
    # for round in range(eps):
    #     for fold in range(k):


@check_shapes("D,D", "P,D", "P,P", "D,A", "P,A", "D", "P")
def bd_loss_k_fold(Kdd, Kpd, Kpp, Kda, Kpa, Yd, Yp, k=10):
    assert len(Kdd) % 2 == 0
    n, eps = len(Kdd) // 2, len(Kpd)
    assert Kda.shape == (2 * n, n)
    assert Kpa.shape == (eps, n)
    # assert np.allclose(Yd, np.hstack([np.full(n, 1), np.full(n, -1)]))
    assert n % k == 0
    m = n // k
    S = jax.lax.dynamic_slice

    # ypred_no_poison = predict_k_fold(Kdd, Yd, k=10)

    def bd_loss_per_fold(fold):
        D = (
            ((np.arange(k) != fold).nonzero(size=k - 1)[0] * m)[np.newaxis].T
            + np.array([0, n])
        ).T.ravel()
        T = fold * m + np.array([0, n])
        P = fold * m
        return bd_loss(
            Kdd=np.block([[S(Kdd, (R, C), (m, m)) for C in D] for R in D]),
            Kpd=np.block([[S(Kpd, (0, C), (eps, m)) for C in D]]),
            Kpp=Kpp,
            Kdt=np.block([[S(Kdd, (R, C), (m, m)) for C in T] for R in D]),
            Kda=np.block([[S(Kda, (R, P), (m, m))] for R in D]),
            Kpt=np.block([[S(Kpd, (0, C), (eps, m)) for C in T]]),
            Kpa=S(Kpa, (0, P), (eps, m)),
            Yd=np.hstack([S(Yd, (C,), (m,)) for C in D]),
            Yp=Yp,
            Yt=np.hstack([S(Yd, (C,), (m,)) for C in T]),
            # Yt=ypred_no_poison[fold],
        )

    result = jax.lax.map(bd_loss_per_fold, np.arange(k))
    return result[0].sum()


bd_loss_k_fold_grad = jit(
    value_and_grad(bd_loss_k_fold, argnums=(1, 2, 4)), backend="cpu"
)


@check_shapes(
    None, "D,D", "D,T", "D,A", "D,H,W,C", "T,H,W,C", "P,H,W,C", "A,H,W,C", "D", "P", "T"
)
def bd_find(kf, Kdd, Kdt, Kda, Xd, Xt, Xp, Xa, Yd, Yp, Yt, kfg=None):
    X = (Xd, Xp, Xt, Xa)
    Kpd, Kpp, Kpt, Kpa = device_put(
        hsplit(keval(kf, Xp, np.vstack(X), batch_size=K_BATCH_SIZE).T, *X), cpu
    )
    (loss, (yt_pred, ya_pred)), (gKpd, gKpp, gKpt, gKpa) = bd_loss_grad(
        Kdd, Kpd, Kpp, Kdt, Kda, Kpt, Kpa, Yd, Yp, Yt
    )
    return (
        loss,
        kgrad_td_rows(
            kfg or kf,
            Xp,
            np.vstack(X),
            np.hstack((gKpd, gKpp + gKpp.T, gKpt, gKpa)),
            wrap=not kfg,
            batch_size=KG_BATCH_SIZE,
        ),
        yt_pred,
        ya_pred,
    )


@check_shapes(None, "D,D", "D,A", "D,H,W,C", "P,H,W,C", "A,H,W,C", "D", "P")
def bd_find_k_fold(kf, Kdd, Kda, Xd, Xp, Xa, Yd, Yp, kfg=None):
    X = (Xd, Xp, Xa)
    Kpd, Kpp, Kpa = device_put(
        hsplit(keval(kf, Xp, np.vstack(X), batch_size=K_BATCH_SIZE).T, *X), cpu
    )
    loss, (gKpd, gKpp, gKpa) = bd_loss_k_fold_grad(Kdd, Kpd, Kpp, Kda, Kpa, Yd, Yp)

    return loss, kgrad_td_rows(
        kfg or kf,
        Xp,
        np.vstack(X),
        np.hstack((gKpd, gKpp + gKpp.T, gKpa)),
        wrap=not kfg,
        batch_size=KG_BATCH_SIZE,
    )


@check_shapes(
    None, "D,D", "D,T", "D,A", "D,H,W,C", "T,H,W,C", "P,H,W,C", "A,H,W,C", "D", "P", "T"
)
def bd_eval(kf, Kdd, Kdt, Kda, Xd, Xt, Xp, Xa, Yd, Yp, Yt):
    X = (Xd, Xp, Xt, Xa)
    Kpd, Kpp, Kpt, Kpa = hsplit(keval(kf, Xp, np.vstack(X)).T, *X)
    yt_pred, ya_pred = vsplit(
        bd_predict(Kdd, Kpd, Kpp, np.hstack([Kdt, Kda]), np.hstack([Kpt, Kpa]), Yd, Yp),
        Kpt.T,
        Kpa.T,
    )

    return yt_pred, ya_pred


# @partial(value_and_grad, argnums=6)
def bd_find_test(kf, Kdd, Kdt, Kda, Xd, Xt, Xp, Xa, Yd, Yp, Yt):
    X = [Xd, Xp, Xt, Xa]
    Kpd, Kpp, Kpt, Kpa = hsplit(keval(kf, Xp, np.vstack(X)).T, *X)
    loss, _, _ = bd_loss(Kdd, Kpd, Kpp, Kdt, Kda, Kpt, Kpa, Yd, Yp, Yt)
    return loss


@partial(value_and_grad, argnums=4)
def bd_find_k_fold_test(kf, Kdd, Kda, Xd, Xp, Xa, Yd, Yp):
    X = (Xd, Xp, Xa)
    Kpd, Kpp, Kpa = device_put(hsplit(keval(kf, Xp, np.vstack(X)).T, *X), cpu)
    print(Kpd, Kpp, Kpa)
    loss = bd_loss_k_fold(Kdd, Kpd, Kpp, Kda, Kpa, Yd, Yp)
    return loss


@check_shapes(
    None, None, "D,D", "D,T", "D,A", "D,H,W,C", "T,H,W,C", "P,H,W,C", "A,H,W,C", "D", "P", "T"
)
def bd_find_fmap(apply_fn, params, Kdd, Kdt, Kda, Xd, Xt, Xp, Xa, Yd, Yp, Yt):
    X = (Xd, Xp, Xt, Xa)
    Kpd, Kpp, Kpt, Kpa = device_put(
        hsplit(keval_fmap(apply_fn, params, Xp, np.vstack(X), batch_size=K_BATCH_SIZE).T, *X), cpu
    )
    (loss, (yt_pred, ya_pred)), (gKpd, gKpp, gKpt, gKpa) = bd_loss_grad(
        Kdd, Kpd, Kpp, Kdt, Kda, Kpt, Kpa, Yd, Yp, Yt
    )
    return (
        loss,
        kgrad_td_rows_fmap2(
            apply_fn,
            params,
            Xp,
            np.vstack(X),
            np.hstack((gKpd, gKpp + gKpp.T, gKpt, gKpa)),
            batch_size=KG_BATCH_SIZE,
        ),
        yt_pred,
        ya_pred,
    )

if __name__ == "__main__":
    print("#" * 80)
    print(f"running on {platform_desc()}")

    model = "convnext2"
    mode = "use-train2"
    eps = 30
    poisoner = "1xs"
    version = "v3"
    version_map = {
        ("wrn34g", "v9.5"): 50,
        ("wrn34g", "v9.3"): 0,
        ("wrn34g", "v9.2"): 100,
        ("wrn34g", "v9"): 500,
        ("wrn34", "v9"): 1500,
        ("wrn34", "v9.2"): 250,
        ("wrn34r", "v9.5"): 50,
        ("wrn34r", "v9.3"): 0,
        ("wrn34r", "v9.2"): 100,
        ("wrn34r", "v9"): 300,
        ("convnext", "v1"): 2000,
        ("convnext", "v1.1"): 500,
        ("convnext", "v1.2"): 0,
        ("convnext", "v1.2"): 0,
        ("convnext", "v3.1"): 1000,
        ("convnext2", "v1"): 50,
        ("convnext2", "v1.0"): 2000,
        ("convnext2", "v1.1"): 0,
        ("convnext2", "v2"): 1500,
        ("convnext2", "v4"): 1000,
        ("convnext2", "v4.1"): 200,
        ("convnext2", "v3"): 3000,
    }

    if model == "convnext2":
        data_file = Path(f"output/X-n04243546-n02096294-{poisoner}-perm.npy")
        n, m = 1300, 50
        s = 2600 if mode == "use-test" else 2200
        t = 100 if mode == "use-test" else 0
        lb = np.tile(datasets.IMAGENET_LOWER_BOUND, (eps, 1, 1, 1))
        ub = np.tile(datasets.IMAGENET_UPPER_BOUND, (eps, 1, 1, 1))

    else:
        data_file = Path(f"output/X-94-{poisoner}-perm.npy")
        n, m = 5000, 1000
        s = 10000 if mode == "use-test" else 8000
        t = 2000 if mode == "use-test" else 0
        lb = np.tile(datasets.CIFAR_LOWER_BOUND, (eps, 1, 1, 1))
        ub = np.tile(datasets.CIFAR_UPPER_BOUND, (eps, 1, 1, 1))

    checkpoint_file = Path(f"output/{model}-{poisoner}/ebd-{version}-{eps}-tr2-init.npy")
    # checkpoint_file = Path(f"output/{model}-{poisoner}/bd-{eps}-tr.npy")
    param_file = Path(
        f"output/checkpoints/{model}-{version.split('.')[0]}/params_{version_map[model, version]:06d}.pkl"
    )
    # param_file = Path(
    #     f"output/checkpoints/0.pkl"
    # )
    kernel_file = Path(f"output/{model}-{poisoner}/gdd-trained-{version}-perm.npy")
    # kernel_file = Path(f"output/{model}-{poisoner}/gdd-perm.npy")
    # kernel_file = Path(f"output/{model}-{poisoner}/gdd-0-perm.npy")


    print(f"     model : {model}")
    print(f"      mode : {mode}")
    print(f"      data : {str(data_file)} ({s}, {t})")
    print(f"checkpoint : {str(checkpoint_file)} ({checkpoint_file.exists()})")
    print(f"    kernel : {str(kernel_file)}")
    print(f"    params : {str(param_file)}")

    if model == "wrn34g":
        ntk = models.WideResnet(block_size=4, k=5, num_classes=1)
    elif model == "wrn34":
        ntk = models.WideResnet(
            block_size=4, k=1, num_classes=1, activation_fn=stax.Relu()
        )
    elif model == "wrn34r":
        ntk = models.WideResnet(
            block_size=4, k=5, num_classes=1, activation_fn=stax.Relu()
        )
    elif model in {"convnext", "convnext2"}:
        ntk = models.ConvNeXt()

    kernel_fn, apply_fn, params = ntk_utils.load_empirical_kernel(
        ntk, param_file, implementation=3
    )
    ekg = ntk_utils.make_empirical_kernel_grad(apply_fn, params)
    kf = kernel_fn

    # kf = partial(ntk[2], get="ntk")
    X = device_put(np.load(data_file), cpu)
    x_train_c, x_train_p, x_train_pp, x_test_c, x_test_p, x_test_pp = vsplit(
        X, n, n, n, m, m, m
    )
    Xd = np.vstack([x_train_c, x_train_pp])
    Yd = np.hstack([np.full(len(x_train_c), 1), np.full(len(x_train_pp), -1)])
    Xp = x_train_p
    Yp = np.full(len(x_train_p), 1)
    Xt = np.vstack([x_test_c, x_test_pp])
    Yt = np.hstack([np.full(len(x_test_c), 1), np.full(len(x_test_pp), -1)])
    Xa = x_test_p
    Ya = np.full(len(x_test_p), 1)

    Yd, Yp, Yt, Ya = device_put((Yd, Yp, Yt, Ya), cpu)
    # Xd, Yd, Xp, Yp, Xt, Yt, Xa, Ya = device_put((Xd, Yd, Xp, Yp, Xt, Yt, Xa, Ya), cpu)
    # G = keval(kf, X, X, batch_size=1000)
    G = np.asarray(np.load(kernel_file))
    G = device_put(G, cpu)
    Gdd, Gpd, Gpp, Gdt, Gpt, Gda, Gpa = bd_split_G(G, n, m)
    delta_bound = np.broadcast_to(8 / 256 / datasets.sigma, (1, 32, 32, 3))

    def kfg(x, y):
        return ekg(x, y)[1]

    # Yt2 = predict2(Gdd, Gdt, Yd)

    # def f(Xp):
    #     return bd_find(kf, Gdd, Gdt, Gda, Xd, Xt, Xp, Xa, Yd, np.ones(len(Xp)), Yt, kfg)

    D = np.s_[(2 * n - s) // 2 : -(2 * n - s) // 2 or None]
    P = np.s_[: -(2 * n - s) // 2 or None]
    T = np.s_[(2 * m - t) // 2 : -(2 * m - t) // 2 or None]
    A = np.s_[(2 * m - t) // 2 :]

    if mode == "use-test":

        def f(xp):
            l, g, yt_pred, ya_pred = bd_find_fmap(
                # kf,
                apply_fn,
                params,
                Gdd[D, D],
                Gdt[D, T],
                Gda[D, A],
                Xd[D],
                Xt[T],
                xp,
                Xa[A],
                Yd[D],
                np.ones(len(xp)),
                Yt[T],
                # kfg,
            )
            return l, g

    elif mode == "use-train":
        assert s < len(Gdd)
        T = (np.s_[: (2 * n - s) // 2], np.s_[-(2 * n - s) // 2 :])
        P = np.s_[: -(2 * n - s) // 2]  # unused
        A = T[0]

        def f(xp):
            l, g, yt_pred, ya_pred = bd_find_fmap(
                # kf,
                apply_fn,
                params,
                Gdd[D, D],
                np.hstack([Gdd[D, C] for C in T]),
                Gpd.T[D, A],
                Xd[D],
                np.vstack([Xd[R] for R in T]),
                xp,
                Xp[A],
                Yd[D],
                np.ones(len(xp)),
                np.hstack([Yd[R] for R in T]),
                # kfg,
            )
            return l, g

    elif mode == "use-train2":
        assert s < len(Gdd)
        T = (np.s_[: (2 * n - s) // 2], np.s_[-(2 * n - s) // 2 :])
        P = np.s_[: -(2 * n - s) // 2]
        A = T[1]

        def f(xp, ret_pred=False):
            l, g, yt_pred, ya_pred = bd_find_fmap(
                # kf,
                apply_fn,
                params,
                Gdd[D, D],
                np.hstack([Gdd[D, C] for C in T]),
                Gpd.T[D, A],
                Xd[D],
                np.vstack([Xd[R] for R in T]),
                xp,
                Xp[A],
                Yd[D],
                np.ones(len(xp)),
                np.hstack([Yd[R] for R in T]),
                # kfg,
            )
            if ret_pred:
                return l,g,yt_pred,ya_pred
            return l, g

    elif mode == "cross-validation":

        def f(xp):
            return bd_find_k_fold(
                kf,
                Gdd[D, D],
                Gpd[D, D].T,
                Xd[D],
                xp,
                Xp[D],
                Yd[D],
                np.ones(len(xp)),
                kfg,
            )

    else:
        raise NotImplementedError

    print(*jax.tree_util.tree_map(str_slice, (D, P, T, A)))

    def bd_callback(xk, _=None):
        if not np.any(np.isnan(xk)):
            np.save(checkpoint_file, xk)
            pass

    def main2():
        if checkpoint_file.exists():
            print("Loading initialization from checkpoint")
            x0 = np.load(checkpoint_file).astype(np.float64)
            # x0 += jax.random.normal(key, x0.shape) * 1e-6

        else:
            print("Computing initialization using greedy algorithm")
            if mode == "use-test":
                x0 = Xp[bd_greedy(Gdd, Gda, Yd, Gpd, Gpa, Gpp, Yp, eps)]
            if mode in ("use-train", "use-train2"):
                # x0 = Xp[
                #     bd_greedy(
                #         Gdd[D, D],
                #         Gpd.T[D, A],
                #         Yd[D],
                #         Gpd[P, D],
                #         Gpp[P, A],
                #         Gpp[P, P],
                #         Yp[P],
                #         eps,
                #     )
                # ]
                trigger = np.tile(np.array([[[1, 1, 1], [-1, -1, -1]],[[-1, -1, -1], [1, 1, 1]]]), (224//2, 224//2, 1))[None]
                x0 = Xd[len(Xd) // 2 :][P][
                    bd_greedy(
                        Gdd[D, D],
                        Gpd.T[D, A],
                        Yd[D],
                        Gpd[P, D],
                        Gpp[P, A],
                        Gpp[P, P],
                        Yp[P],
                        eps,
                    )
                ] + ((-1) ** np.arange(10))[:, None, None, None] * trigger/16
            else:
                raise NotImplementedError
                x0 = Xp[bd_greedy(Gdd, Gpd.T, Yd, Gpd, Gpp, Gpp, Yp, eps)]
        # x0 = Xp[:eps]
        # x0 = jax.random.normal(key, x0.shape)

        print("Starting sp_minimize")
        opt_min = sp_minimize(
            f,
            x0,
            bounds=(lb, ub),
            callback=bd_callback,
            method="L-BFGS-B",
            options={
                "iprint": 1,
                "maxcor": 100,
            },
        )

        # import optax
        # from jax_utils import optax_minimize
        # print("Starting optax_minimize")
        # opt_min = optax_minimize(
        #     f,
        #     x0,
        #     optax.sgd(1e-5, momentum=0.90),
        #     bounds=(lb, ub),
        #     callback=bd_callback,
        # )

        return opt_min

    # main2()
