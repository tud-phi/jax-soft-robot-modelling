import dill
from pathlib import Path
import sympy as sp
from typing import Callable, Dict, Tuple, Union

from .symbolic_utils import compute_coriolis_matrix


def symbolically_derive_pendulum_model(
        num_links: int, filepath: Union[str, Path] = None
) -> Dict:
    """
    Symbolically derive the kinematics and dynamics of a n-link pendulum.
    We use the relative joint angles between links as the generalized coordinates.
    Args:
        num_links: number of pendulum links
        filepath: path to save the derived model
    """
    m_syms = sp.symbols(f"m1:{num_links + 1}")  # mass of each link
    I_syms = sp.symbols(f"I1:{num_links + 1}")  # moment of inertia of each link
    l_syms = sp.symbols(f"l1:{num_links + 1}")  # length of each link
    lc_syms = sp.symbols(f"lc1:{num_links + 1}")  # center of mass of each link (distance from joint)
    g_syms = sp.symbols(f"g1:3")  # gravity vector

    # configuration variables and their derivatives
    q_syms = sp.symbols(f"q1:{num_links + 1}")  # joint angle
    q_d_syms = sp.symbols(f"q_d1:{num_links + 1}")  # joint velocity

    # construct the symbolic matrices
    m = sp.Matrix(m_syms)  # mass of each link
    I = sp.Matrix(I_syms)  # moment of inertia of each link
    l = sp.Matrix(l_syms)  # length of each link
    lc = sp.Matrix(lc_syms)  # center of mass of each link (distance from joint)
    g = sp.Matrix(g_syms)  # gravity vector

    # configuration variables and their derivatives
    q = sp.Matrix(q_syms)  # joint angle
    q_d = sp.Matrix(q_d_syms)  # joint velocity

    # orientation scalar and rotation matrix
    th_ls, R_ls = [], []
    # positions of tip of link and center of mass respectively
    x_ls, xc_ls = [], []
    # positional Jacobians of tip of link and center of mass respectively
    Jx_ls, Jxc_ls = [], []
    # orientation Jacobian
    Jo_ls = []
    # mass matrix
    B = sp.zeros(num_links, num_links)
    # potential energy
    U = sp.Matrix([[0]])

    # initialize
    th_prev = 0.0
    x_prev = sp.Matrix([0, 0])
    for i in range(num_links):
        # orientation of link
        th = th_prev + q[i]
        th_ls.append(th)

        # absolute rotation of link
        R = sp.Matrix([
            [sp.cos(th), -sp.sin(th)],
            [sp.sin(th), sp.cos(th)]]
        )
        R_ls.append(R)

        # absolute position of center of mass
        xc = sp.simplify(x_prev + R @ sp.Matrix([lc[i], 0]))
        xc_ls.append(xc)

        # absolute position of end of link
        x = sp.simplify(x_prev + R @ sp.Matrix([l[i], 0]))
        x_ls.append(x)

        # positional Jacobian of end of link
        Jx = sp.simplify(x.jacobian(q))
        Jx_ls.append(Jx)

        # positional Jacobian of center of mass
        Jxc = sp.simplify(xc.jacobian(q))
        Jxc_ls.append(Jxc)

        # orientation Jacobian
        Jo = sp.simplify(sp.Matrix([[th]]).jacobian(q))
        Jo_ls.append(Jo)

        # add to mass matrix
        B = B + sp.simplify(m[i] * Jxc.T @ Jxc + I[i] * Jo.T @ Jo)

        # add to potential energy
        U = U + sp.simplify(m[i] * g.T @ xc)

        # update for next iteration
        th_prev = th_ls[i]
        x_prev = x_ls[i]

    # simplify mass matrix
    B = sp.simplify(B)
    print("B =\n", B)

    C = compute_coriolis_matrix(B, q, q_d)
    print("C =\n", C)

    # compute the gravity force vector
    G = sp.simplify(- U.jacobian(q).transpose())
    print("G =\n", G)

    # dictionary with functions
    sym_exps = {
        "params_syms": {
            "m": m_syms,
            "I": I_syms,
            "l": l_syms,
            "lc": lc_syms,
            "g": g_syms,
        },
        "state_syms": {
            "q": q_syms,
            "q_d": q_d_syms,
        },
        "exps": {
            "B": B,
            "C": C,
            "G": G,
        }
    }

    if filepath is not None:
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)

        with open(str(filepath), "wb") as f:
            dill.dump(sym_exps, f)

    return sym_exps
