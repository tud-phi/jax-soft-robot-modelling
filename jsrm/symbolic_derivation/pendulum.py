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
    Returns:
        sym_exps: dictionary with entries
            params_syms: dictionary of robot parameters
            state_syms: dictionary of state variables
            exps: dictionary of symbolic expressions
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
    # matrix with tip of link and center of mass positions
    p_mx, pc_mx = sp.zeros(2, num_links), sp.zeros(2, num_links)
    # positional Jacobians of tip of link and center of mass respectively
    Jp_ls, Jpc_ls = [], []
    # orientation Jacobian
    Jo_ls = []
    # mass matrix
    B = sp.zeros(num_links, num_links)
    # potential energy
    U = sp.Matrix([[0]])

    # initialize
    th_prev = 0.0
    p_prev = sp.Matrix([0, 0])
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
        pc = sp.simplify(p_prev + R @ sp.Matrix([lc[i], 0]))
        pc_mx[:, i] = pc

        # absolute position of end of link
        p = sp.simplify(p_prev + R @ sp.Matrix([l[i], 0]))
        p_mx[:, i] = p

        # positional Jacobian of end of link
        Jp = sp.simplify(p.jacobian(q))
        Jp_ls.append(Jp)

        # positional Jacobian of center of mass
        Jpc = sp.simplify(pc.jacobian(q))
        Jpc_ls.append(Jpc)

        # orientation Jacobian
        Jo = sp.simplify(sp.Matrix([[th]]).jacobian(q))
        Jo_ls.append(Jo)

        # add to mass matrix
        B = B + sp.simplify(m[i] * Jpc.T @ Jpc + I[i] * Jo.T @ Jo)

        # add to potential energy
        U = U + sp.simplify(m[i] * g.T @ pc)

        # update for next iteration
        th_prev = th_ls[i]
        p_prev = p

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
            "p": p_mx,
            "pc": pc_mx,
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
