import dill
from pathlib import Path
import sympy as sp
from typing import Callable, Dict, Optional, Tuple, Union

from .symbolic_utils import compute_coriolis_matrix, compute_dAdt


def symbolically_derive_planar_pcs_model(
    num_segments: int,
    filepath: Optional[Union[str, Path]] = None,
    simplify_expressions: bool = True,
) -> Dict:
    """
    Symbolically derive the kinematics and dynamics of a planar continuum soft robot modelled with
    Piecewise Constant Strain.
    Args:
        num_segments: number of constant strain segments
        filepath: path to save the derived model
        simplify_expressions: if true, simplify the expressions (might take some time). Default is True.
    Returns:
        sym_exps: dictionary with entries
            params_syms: dictionary of robot parameters
            state_syms: dictionary of state variables
            exps: dictionary of symbolic expressions
    """
    # number of degrees of freedom
    num_dof = (
        3 * num_segments
    )  # we allow for 3 strains for each segment (bending, shear, elongation)

    th0 = sp.Symbol("th0", real=True)  # initial angle of the robot
    rho_syms = list(
        sp.symbols(f"rho1:{num_segments + 1}", nonnegative=True)
    )  # volumetric mass density [kg/m^3]
    l_syms = list(
        sp.symbols(f"l1:{num_segments + 1}", nonnegative=True, nonzero=True)
    )  # length of each segment [m]
    r_syms = list(
        sp.symbols(f"r1:{num_segments + 1}", nonnegative=True)
    )  # radius of each segment [m]
    g_syms = list(sp.symbols(f"g1:3"))  # gravity vector

    # planar strains and their derivatives
    xi_syms = list(sp.symbols(f"xi1:{num_dof + 1}", nonzero=True))  # strains
    xi_d_syms = list(sp.symbols(f"xi_d1:{num_dof + 1}"))  # strain time derivatives

    # construct the symbolic matrices
    rho = sp.Matrix(rho_syms)  # volumetric mass density [kg/m^3]
    l = sp.Matrix(l_syms)  # length of each link
    r = sp.Matrix(r_syms)  # radius of segment
    g = sp.Matrix(g_syms)  # gravity vector

    # configuration variables and their derivatives
    xi = sp.Matrix(xi_syms)  # strains
    xi_d = sp.Matrix(xi_d_syms)  # strain time derivatives

    # matrix with symbolic expressions to derive the poses along each segment
    chi_sms = []
    # Jacobians (positional + orientation) in each segment as a function of the point coordinate s and its time derivative
    J_sms, J_d_sms = [], []
    # cross-sectional area of each segment
    A = sp.zeros(num_segments)
    # second area moment of inertia of each segment
    I = sp.zeros(num_segments)
    # inertia matrix
    B = sp.zeros(num_dof, num_dof)
    # potential energy
    U_g = sp.Matrix([[0]])

    # symbol for the point coordinate
    s = sp.symbols("s", real=True, nonnegative=True)

    # initialize
    th_prev = th0
    p_prev = sp.Matrix([0, 0])
    for i in range(num_segments):
        # bending strain
        kappa = xi[3 * i]
        # shear strain
        sigma_x = xi[3 * i + 1]
        # axial strain
        sigma_y = xi[3 * i + 2]

        # compute the cross-sectional area of the rod
        A[i] = sp.pi * r[i] ** 2

        # compute the second area moment of inertia of the rod
        I[i] = A[i] ** 2 / (4 * sp.pi)

        # planar orientation of robot as a function of the point s
        th = th_prev + s * kappa

        # absolute rotation of link
        R = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])

        # derivative of Cartesian position as function of the point s
        dp_ds = R @ sp.Matrix([sigma_x, sigma_y])

        # position along the current rod as a function of the point s
        p = p_prev + sp.integrate(dp_ds, (s, 0.0, s))

        # symbolic expression for the pose in the current segment as a function of the point s
        chi = sp.zeros(3, 1)
        chi[:2, 0] = p  # the x and y position
        chi[2, 0] = th  # the orientation angle theta
        chi_sms.append(chi)

        print(f"chi of segment {i+1}:\n", chi)

        # positional Jacobian as a function of the point s
        Jp = sp.simplify(p.jacobian(xi))

        # orientation Jacobian
        Jo = sp.simplify(sp.Matrix([[th]]).jacobian(xi))

        # combine positional and orientation Jacobian
        # the first two rows correspond to the position and the last row to the orientation
        # the columns correspond to the strains xi
        J = Jp.col_join(Jo)
        J_sms.append(J)

        # compute the time derivative of the Jacobian
        J_d = compute_dAdt(J, xi, xi_d)  # time derivative of the end-effector Jacobian
        J_d_sms.append(J_d)

        # derivative of mass matrix with respect to the point coordinate s
        dB_ds = rho[i] * A[i] * Jp.T @ Jp + rho[i] * I[i] * Jo.T @ Jo
        if simplify_expressions:
            dB_ds = sp.simplify(dB_ds)
        # mass matrix of the current segment
        B_i = sp.integrate(dB_ds, (s, 0, l[i]))
        if simplify_expressions:
            B_i = sp.simplify(B_i)
        # add mass matrix of segment to previous segments
        B = B + B_i

        # derivative of the potential energy with respect to the point coordinate s
        dU_g_ds = -rho[i] * A[i] * g.T @ p
        if simplify_expressions:
            dU_g_ds = sp.simplify(dU_g_ds)
        # gravitational potential energy of the current segment
        U_gi = sp.integrate(dU_g_ds, (s, 0, l[i]))
        if simplify_expressions:
            U_gi = sp.simplify(U_gi)
        # add potential energy of segment to previous segments
        U_g = U_g + U_gi

        # update the orientation for the next segment
        th_prev = th.subs(s, l[i])

        # update the position for the next segment
        p_prev = p.subs(s, l[i])

    if simplify_expressions:
        # simplify mass matrix
        B = sp.simplify(B)
    print("B =\n", B)

    C = compute_coriolis_matrix(B, xi, xi_d, simplify=simplify_expressions)
    print("C =\n", C)

    # compute the gravity force vector
    G = U_g.jacobian(xi).transpose()
    if simplify_expressions:
        G = sp.simplify(G)
    print("G =\n", G)

    # dictionary with expressions
    sym_exps = {
        "params_syms": {
            "th0": th0,
            "l": l_syms,
            "r": r_syms,
            "rho": rho_syms,
            "g": g_syms,
        },
        "state_syms": {
            "xi": xi_syms,
            "xi_d": xi_d_syms,
            "s": s,
        },
        "exps": {
            "chi_sms": chi_sms,  # list of pose expressions (for each segment)
            "chiee": chi_sms[-1].subs(
                s, l[-1]
            ),  # expression for end-effector pose of shape (3, )
            "J_sms": J_sms,  # list of Jacobians (for each segment) of shape (3, num_dof)
            "Jee": J_sms[-1].subs(
                s, l[-1]
            ),  # end-effector Jacobian of shape (3, num_dof)
            "J_d_sms": J_d_sms,  # list of time derivatives of Jacobians (for each segment)
            "Jee_d": J_d_sms[-1].subs(
                s, l[-1]
            ),  # time derivative of end-effector Jacobian of shape (3, num_dof)
            "B": B,  # mass matrix
            "C": C,  # coriolis matrix
            "G": G,  # gravity vector
            "U_g": U_g,  # gravitational potential energy
        },
    }

    if filepath is not None:
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)

        with open(str(filepath), "wb") as f:
            dill.dump(sym_exps, f)

    return sym_exps
