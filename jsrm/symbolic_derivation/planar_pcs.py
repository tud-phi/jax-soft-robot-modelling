import dill
from pathlib import Path
import sympy as sp
from typing import Callable, Dict, Tuple, Union

from .symbolic_utils import compute_coriolis_matrix


def symbolically_derive_planar_pcs_model(
        num_segments: int, filepath: Union[str, Path] = None
) -> Dict:
    """
    Symbolically derive the kinematics and dynamics of a planar continuum soft robot modelled with
    Piecewise Constant Strain.
    Args:
        num_segments: number of constant strain segments
        filepath: path to save the derived model
    Returns:
        sym_exps: dictionary with entries
            params_syms: dictionary of robot parameters
            state_syms: dictionary of state variables
            exps: dictionary of symbolic expressions
    """
    # number of degrees of freedom
    num_dof = 3 * num_segments  # we allow for 3 strains for each segment (bending, shear, elongation)

    rho_syms = list(sp.symbols(f"rho1:{num_segments + 1}", nonnegative=True))  # surface mass density [kg/m^2]
    l_syms = list(sp.symbols(f"l1:{num_segments + 1}", nonnegative=True, nonzero=True))  # length of each segment [m]
    r_syms = list(sp.symbols(f"r1:{num_segments + 1}", nonnegative=True))  # radius of each segment [m]
    g_syms = list(sp.symbols(f"g1:3"))  # gravity vector

    # planar strains and their derivatives
    xi_syms = list(sp.symbols(f"xi1:{num_dof + 1}", nonzero=True))  # strains
    xi_d_syms = list(sp.symbols(f"xi_d1:{num_dof + 1}"))  # strain time derivatives

    # construct the symbolic matrices
    rho = sp.Matrix(rho_syms)  # surface mass density [kg/m^2]
    l = sp.Matrix(l_syms)  # length of each link
    r = sp.Matrix(r_syms)   # radius of segment
    g = sp.Matrix(g_syms)  # gravity vector

    # configuration variables and their derivatives
    xi = sp.Matrix(xi_syms)  # strains
    xi_d = sp.Matrix(xi_d_syms)  # strain time derivatives

    # orientation scalar and rotation matrix
    th_ls, R_ls = [], []
    # matrix with symbolic expressions to derive the positions along each segment
    p_mx = sp.zeros(2, num_segments)
    # positional Jacobians of tip of link and center of mass respectively
    Jp_ls, Jpc_ls = [], []
    # orientation Jacobian
    Jo_ls = []
    # linear inertia distribution for each segment
    I = sp.zeros(num_segments)
    # inertia matrix
    B = sp.zeros(num_dof, num_dof)
    # potential energy
    U = sp.Matrix([[0]])

    # symbol for the point coordinate
    s = sp.symbols("s", real=True, nonnegative=True)
    # symbol for integration of inertia distribution
    _r = sp.symbols("_r")

    # initialize
    th_prev = 0.0
    p_prev = sp.Matrix([0, 0])
    for i in range(num_segments):
        # bending strain
        kappa = xi[3 * i]
        # shear strain
        sigma_x = xi[3 * i + 1]
        # elongation strain
        sigma_y = xi[3 * i + 2]

        # linear mass density [kg / m]
        lambda_i = 2 * r[i] * rho[i]

        # compute the inertia distribution
        I[i] = sp.integrate(rho[i] * _r ** 2, (_r, -r[i], r[i]))

        # planar orientation of robot as a function of the point s
        th = th_prev + s * kappa

        # absolute rotation of link
        R = sp.Matrix([
            [sp.cos(th), -sp.sin(th)],
            [sp.sin(th), sp.cos(th)]]
        )
        R_ls.append(R)

        # derivative of Cartesian position as function of the point s
        dp_ds = R @ sp.Matrix([sigma_x, sigma_y])

        # position along the current rod as a function of the point s
        p = p_prev + sp.integrate(dp_ds, (s, 0.0, s))
        p_mx[:, i] = p

        print(f"p of segment {i+1}:\n", p)

        # positional Jacobian as a function of the point s
        Jp = sp.simplify(p.jacobian(xi))
        Jp_ls.append(Jp)

        # orientation Jacobian
        Jo = sp.simplify(sp.Matrix([[th]]).jacobian(xi))
        Jo_ls.append(Jo)

        # derivative of mass matrix with respect to the point coordinate s
        dB_ds = sp.simplify(lambda_i * Jp.T @ Jp + I[i] * Jo.T @ Jo)
        # mass matrix of the current segment
        B_i = sp.simplify(sp.integrate(dB_ds, (s, 0, l[i])))
        # add mass matrix of segment to previous segments
        B = B + B_i

        # derivative of the potential energy with respect to the point coordinate s
        dU_ds = sp.simplify(lambda_i * g.T @ p)
        # potential energy of the current segment
        U_i = sp.simplify(sp.integrate(dU_ds, (s, 0, l[i])))
        # add potential energy of segment to previous segments
        U = U + U_i

        # update the orientation for the next segment
        th_ls.append(th.subs(s, l[i]))
        th_prev = th_ls[i]

        # update the position for the next segment
        p_prev = p.subs(s, l[i])

    # simplify mass matrix
    B = sp.simplify(B)
    print("B =\n", B)

    C = compute_coriolis_matrix(B, xi, xi_d)
    print("C =\n", C)

    # compute the gravity force vector
    G = sp.simplify(- U.jacobian(xi).transpose())
    print("G =\n", G)

    # dictionary with functions
    sym_exps = {
        "params_syms": {
            "rho": rho_syms,
            "l": l_syms,
            "g": g_syms,
        },
        "state_syms": {
            "xi": xi_syms,
            "xi_d": xi_d_syms,
        },
        "exps": {
            "p": p_mx,  # matrix with positions as a function
            "pee": p_mx.subs(s, l[-1]),  # vector of shape (2, ) with end-effector position
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