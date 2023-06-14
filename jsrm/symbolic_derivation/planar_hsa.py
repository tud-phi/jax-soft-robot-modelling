import dill
from pathlib import Path
import sympy as sp
from typing import Callable, Dict, Tuple, Union

from .symbolic_utils import compute_coriolis_matrix


def symbolically_derive_planar_hsa_model(
    num_segments: int, filepath: Union[str, Path] = None, th0: float = 0.0
) -> Dict:
    """
    Symbolically derive the kinematics and dynamics of a planar hsa robot modelled with
    Piecewise Constant Strain.
    Args:
        num_segments: number of constant strain segments
        filepath: path to save the derived model
        th0: initial angle of the robot
    Returns:
        sym_exps: dictionary with entries
            params_syms: dictionary of robot parameters
            state_syms: dictionary of state variables
            exps: dictionary of symbolic expressions
    """
    # number of degrees of freedom
    num_dof = (
        4 * num_segments
    )  # we allow for 4 strains for each segment (bending, twist, shear, elongation)

    rho_syms = list(
        sp.symbols(f"rho1:{num_segments + 1}", nonnegative=True)
    )  # volumetric mass density [kg/m^3]
    l_syms = list(
        sp.symbols(f"l1:{num_segments + 1}", nonnegative=True, nonzero=True)
    )  # length of each segment [m]
    rin_syms = list(
        sp.symbols(f"rin1:{num_segments + 1}", nonnegative=True)
    )  # inner radius of each segment [m]
    rout_syms = list(
        sp.symbols(f"rout1:{num_segments + 1}", nonnegative=True)
    )  # outside radius of each segment [m]
    g_syms = list(sp.symbols(f"g1:3"))  # gravity vector
    roff_syms = list(
        sp.symbols(f"roff1:{num_segments + 1}", nonnegative=True)
    )  # radial offset of left and right rod from virtual rod in the center

    # planar strains and their derivatives
    xi_syms = list(sp.symbols(f"xi1:{num_dof + 1}", nonzero=True))  # strains
    xi_d_syms = list(sp.symbols(f"xi_d1:{num_dof + 1}"))  # strain time derivatives

    # construct the symbolic matrices
    rho = sp.Matrix(rho_syms)  # volumetric mass density [kg/m^3]
    l = sp.Matrix(l_syms)  # length of each link
    rin = sp.Matrix(rin_syms)  # inside radius of segment
    rout = sp.Matrix(rout_syms)  # outside radius of segment
    g = sp.Matrix(g_syms)  # gravity vector
    roff = sp.Matrix(roff_syms)

    # configuration variables and their derivatives
    xi = sp.Matrix(xi_syms)  # strains
    xi_d = sp.Matrix(xi_d_syms)  # strain time derivatives

    # matrix with symbolic expressions to derive the poses along each segment
    chi_sms = []
    # Jacobians (positional + orientation) in each segment as a function of the point coordinate s
    J_sms = []
    # cross-sectional area of each segment
    A = sp.zeros(num_segments)
    # second area moment of inertia of each segment
    I = sp.zeros(num_segments)
    # inertia matrix
    B = sp.zeros(num_dof, num_dof)
    # potential energy
    U = sp.Matrix([[0]])

    # symbol for the point coordinate
    s = sp.symbols("s", real=True, nonnegative=True)

    # initialize
    th_prev = th0
    p_prev = sp.Matrix([0, 0])
    for i in range(num_segments):
        # bending strain
        kappa = xi[3 * i]
        # shear strain
        sigma_sh = xi[3 * i + 1]
        # axial strain
        sigma_a = xi[3 * i + 2]

        # compute the cross-sectional area of the rod
        A[i] = sp.pi * (rout[i] ** 2 - rin[i] ** 2)

        # compute the second area moment of inertia of the rod
        I[i] = sp.pi / 4 * (rout[i] ** 4 - rin[i] ** 4)

        # planar orientation of robot as a function of the point s
        th = th_prev + s * kappa

        # absolute rotation of link
        R = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])

        # derivative of Cartesian position as function of the point s
        dp_ds = R @ sp.Matrix([sigma_sh, sigma_a])

        # position along the virtual center rod as a function of the point s
        p = p_prev + sp.integrate(dp_ds, (s, 0.0, s))

        # position along the left rod as a function of the point s
        p_left = p + R @ sp.Matrix([-roff[i], 0.0])
        p_right = p + R @ sp.Matrix([roff[i], 0.0])

        # symbolic expression for the pose in the current segment as a function of the point s
        chi = sp.zeros(3, 1)
        chi[:2, 0] = p  # the x and y position
        chi[2, 0] = th  # the orientation angle theta
        chi_sms.append(chi)

        print(f"chi of segment {i+1}:\n", chi)

        # positional Jacobian as a function of the point s
        Jp = sp.simplify(p.jacobian(xi))
        Jp_left = sp.simplify(p_left.jacobian(xi))
        Jp_right = sp.simplify(p_right.jacobian(xi))

        # orientation Jacobian
        Jo = sp.simplify(sp.Matrix([[th]]).jacobian(xi))

        # combine positional and orientation Jacobian
        # the first two rows correspond to the position and the last row to the orientation
        # the columns correspond to the strains xi
        J = Jp.col_join(Jo)
        J_sms.append(J)

        # derivative of mass matrix with respect to the point coordinate s
        # TODO: integrate along radius of rod or shift second moment of inertia?
        dB_ds = sp.simplify(
            rho[i] * A[i] * (Jp_left.T @ Jp_left + Jp_right.T @ Jp_right)
            + rho[i] * I[i] * Jo.T @ Jo
        )
        # mass matrix of the current segment
        B_i = sp.simplify(sp.integrate(dB_ds, (s, 0, l[i])))
        # add mass matrix of segment to previous segments
        B = B + B_i

        # derivative of the potential energy with respect to the point coordinate s
        dU_ds = sp.simplify(rho[i] * A[i] * g.T @ (p_left + p_right))
        # potential energy of the current segment
        U_i = sp.simplify(sp.integrate(dU_ds, (s, 0, l[i])))
        # add potential energy of segment to previous segments
        U = U + U_i

        # update the orientation for the next segment
        th_prev = th.subs(s, l[i])

        # update the position for the next segment
        p_prev = p.subs(s, l[i])

    # simplify mass matrix
    B = sp.simplify(B)
    print("B =\n", B)

    C = compute_coriolis_matrix(B, xi, xi_d)
    print("C =\n", C)

    # compute the gravity force vector
    G = sp.simplify(-U.jacobian(xi).transpose())
    print("G =\n", G)

    # dictionary with expressions
    sym_exps = {
        "params_syms": {
            "rho": rho_syms,
            "l": l_syms,
            "r": r_syms,
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
            "J_sms": J_sms,
            "Jee": J_sms[-1].subs(s, l[-1]),
            "B": B,
            "C": C,
            "G": G,
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
