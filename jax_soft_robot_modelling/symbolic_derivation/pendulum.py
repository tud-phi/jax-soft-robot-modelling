import numpy as np
from pathlib import Path
import sympy as sp
from typing import Union


def symbolically_derive_pendulum_model(num_links: int, filepath: Union[str, Path] = None):
    m = sp.Matrix(sp.symbols(f"m1:{num_links + 1}"))  # mass of each link
    I = sp.Matrix(sp.symbols(f"I1:{num_links + 1}"))  # moment of inertia of each link
    l = sp.Matrix(sp.symbols(f"l1:{num_links + 1}"))  # length of each link
    lc = sp.Matrix(sp.symbols(f"lc1:{num_links + 1}"))  # center of mass of each link (distance from joint)

    # parameters for potential energy
    g = sp.Matrix(sp.symbols(f"g1:3"))  # gravity vector

    q = sp.Matrix(sp.symbols(f"q1:{num_links + 1}"))  # joint angle
    q_d = sp.Matrix(sp.symbols(f"q_d1:{num_links + 1}"))  # joint velocity

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

    # compute the Christoffel symbols
    Ch_flat = []
    for i in range(num_links):
        for j in range(num_links):
            for k in range(num_links):
                # Ch[i, j, k] = sp.simplify(0.5 * (B[i, j].diff(q[k]) + B[i, k].diff(q[j]) - B[j, k].diff(q[i])))
                Ch_ijk =0.5 * (B[i, j].diff(q[k]) + B[i, k].diff(q[j]) - B[j, k].diff(q[i]))
                Ch_flat.append(sp.simplify(Ch_ijk, rational=True))
    Ch = sp.Array(Ch_flat, (num_links, num_links, num_links))

    # compute the coriolis and centrifugal force matrix
    C = sp.zeros(num_links, num_links)
    for i in range(num_links):
        for j in range(num_links):
            for k in range(num_links):
                C[i, j] = C[i, j] + Ch[i, j, k] * q_d[k]
    # simplify coriolis and centrifugal force matrix
    C = sp.simplify(C, rational=True)
    print("C =\n", C)

    # compute the gravity force vector
    G = sp.simplify(- U.jacobian(q).transpose())
    print("G =\n", G)

    if filepath is not None:
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)

