import dill
from pathlib import Path
import sympy as sp
from typing import Callable, Dict, Tuple, Union

from .symbolic_utils import compute_coriolis_matrix, compute_dAdt


def symbolically_derive_planar_hsa_model(
    num_segments: int,
    filepath: Union[str, Path] = None,
    num_rods_per_segment: int = 2,
    simplify: bool = True,
) -> Dict:
    """
    Symbolically derive the kinematics and dynamics of a planar hsa robot modelled with
    Piecewise Constant Strain.
    Args:
        num_segments: number of constant strain segments
        filepath: path to save the derived model
        num_rods_per_segment: number of HSA rods per segment
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
    l_syms = list(
        sp.symbols(f"l1:{num_segments + 1}", nonnegative=True, nonzero=True)
    )  # length of each segment [m]
    lpc_syms = list(
        sp.symbols(f"lpc1:{num_segments + 1}", nonnegative=True, nonzero=True)
    )  # length of the rigid proximal caps of the rods connecting to the base [m]
    ldc_syms = list(
        sp.symbols(f"ldc1:{num_segments + 1}", nonnegative=True, nonzero=True)
    )  # length of the rigid distal caps of the rods connecting to the platform [m]
    rout_syms = list(
        sp.symbols(f"rout1:{num_segments * num_rods_per_segment + 1}", nonnegative=True)
    )  # outside radius of each segment [m]
    rin_syms = list(
        sp.symbols(f"rin1:{num_segments * num_rods_per_segment + 1}", nonnegative=True)
    )  # inner radius of each segment [m]
    roff_syms = list(
        sp.symbols(f"roff1:{num_segments * num_rods_per_segment + 1}", nonnegative=True)
    )  # radial offset of each rod from the centerline
    pcudim_syms = list(
        sp.symbols(f"pcudim1:{3*num_segments + 1}", nonnegative=True)
    )  # dimensions of platform cuboid consisting of [width, height, depth] [m]
    rhor_syms = list(
        sp.symbols(f"rhor1:{num_segments * num_rods_per_segment + 1}", nonnegative=True)
    )  # volumetric mass density of the rods [kg/m^3]
    rhop_syms = list(
        sp.symbols(f"rhop1:{num_segments + 1}", nonnegative=True)
    )  # volumetric mass density of the platform [kg/m^3]
    rhoec_syms = list(
        sp.symbols(f"rhoec1:{num_segments + 1}", nonnegative=True)
    )  # volumetric mass density of the rod end caps (both at the proximal and distal ends) [kg/m^3]
    g_syms = list(sp.symbols(f"g1:3"))  # gravity vector

    # planar strains and their derivatives
    xi_syms = list(sp.symbols(f"xi1:{num_dof + 1}", nonzero=True))  # strains
    xi_d_syms = list(sp.symbols(f"xi_d1:{num_dof + 1}"))  # strain time derivatives

    # construct the symbolic matrices
    l = sp.Matrix(l_syms)  # length of each segment
    lpc = sp.Matrix(
        [lpc_syms]
    )  # length of the rigid proximal caps of the rods connecting to the base
    ldc = sp.Matrix(
        [ldc_syms]
    )  # length of the rigid distal end caps of the rods connecting to the platform
    rout = sp.Matrix(rout_syms).reshape(
        num_segments, num_rods_per_segment
    )  # outside radius of each rod
    rin = sp.Matrix(rin_syms).reshape(
        num_segments, num_rods_per_segment
    )  # inside radius of each rod
    # radial offset of each rod from the centerline
    roff = sp.Matrix(roff_syms).reshape(num_segments, num_rods_per_segment)
    # dimensions of platform cuboid consisting of [width, height, depth] [m]
    pcudim = sp.Matrix(pcudim_syms).reshape(num_segments, 3)
    # volumetric mass density of the rods [kg/m^3]
    rhor = sp.Matrix(rhor_syms).reshape(num_segments, num_rods_per_segment)
    # volumetric mass density of the platform [kg/m^3]
    rhop = sp.Matrix(rhop_syms)
    # volumetric mass density of the rod end caps (both at the proximal and distal ends) [kg/m^3]
    rhoec = sp.Matrix(rhoec_syms)
    g = sp.Matrix(g_syms)  # gravity vector

    # configuration variables and their derivatives
    xi = sp.Matrix(xi_syms)  # strains
    xi_d = sp.Matrix(xi_d_syms)  # strain time derivatives

    # matrix with symbolic expressions to derive the poses along the centerline of each segment
    chiv_sms = []
    # Jacobians (positional + orientation) in each segment as a function of the point coordinate s
    Jv_sms = []
    # poses and their Jacobian along each rod
    chir_sms, Jr_sms = [], []
    # poses and their Jacobian of each platform CoG
    chip_sms, Jp_sms = [], []

    # cross-sectional area of each rod
    Ar = sp.zeros(num_segments, num_rods_per_segment)
    # second area moment of inertia for bending of each rod
    Ir = sp.zeros(num_segments, num_rods_per_segment)
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

        # planar orientation of robot as a function of the point s
        th = th_prev + s * kappa

        # absolute rotation of link
        R = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])

        # derivative of Cartesian position as function of the point s
        dp_ds = R @ sp.Matrix([sigma_sh, sigma_a])

        # position along the virtual center rod as a function of the point s
        p = p_prev + sp.Matrix([0, lpc[i]]) + sp.integrate(dp_ds, (s, 0.0, s))

        # symbolic expression for the pose of the virtual rod at the centerline as a function of the point s
        chiv = sp.zeros(3, 1)
        chiv[:2, 0] = p  # the x and y position
        chiv[2, 0] = th  # the orientation angle theta
        chiv_sms.append(chiv)

        # positional Jacobian as a function of the point s
        Jvp = sp.simplify(p.jacobian(xi))  # orientation Jacobian
        Jvo = sp.simplify(sp.Matrix([[th]]).jacobian(xi))

        # combine positional and orientation Jacobian
        # the first two rows correspond to the position and the last row to the orientation
        # the columns correspond to the strains xi
        Jv = Jvp.col_join(Jvo)
        Jv_sms.append(Jv)

        for j in range(num_rods_per_segment):
            # compute the cross-sectional area of the rod
            Ar[i, j] = sp.pi * (rout[i, j] ** 2 - rin[i, j] ** 2)
            # compute the second area moment of inertia of the rod
            Ir[i, j] = sp.pi / 4 * (rout[i, j] ** 4 - rin[i, j] ** 4)

            pr = p + R @ sp.Matrix([roff[i, j], 0.0])
            chir = sp.zeros(3, 1)
            chir[:2, 0] = pr  # the x and y position
            chir[2, 0] = th  # the orientation angle theta
            chir_sms.append(chir)

            # Jacobian of rod poses with respect to strains of virtual center rod
            Jr = chiv.jacobian(xi)
            Jrp = pr.jacobian(xi)  # positional Jacobian
            Jr_sms.append(Jr)

            # integrate mass matrix of each rod
            dBr_ds = sp.simplify(
                rhor[i, j] * Ar[i, j] * Jrp.T @ Jrp
                + rhor[i, j] * Ir[i, j] * Jvo.T @ Jvo
            )
            # mass matrix of the current rod
            Br_ij = sp.integrate(dBr_ds, (s, 0, l[i]))
            if simplify:
                Br_ij = sp.simplify(Br_ij)
            # add the mass matrix
            B = B + Br_ij

            # derivative of the potential energy with respect to the point coordinate s
            dUr_ds = sp.simplify(rhor[i, j] * Ar[i, j] * g.T @ pr)
            # potential energy of the current segment
            U_ij = sp.simplify(sp.integrate(dUr_ds, (s, 0, l[i])))
            # add potential energy of segment to previous segments
            U = U + U_ij

        # mass of the platform itself
        mp_itself = rhop[i, 0] * pcudim[i, 0] * pcudim[i, 1] * pcudim[i, 2]
        # inertia of the platform itself about its own CoG
        Ip_itself = mp_itself / 12 * (pcudim[i, 0] ** 2 + pcudim[i, 1] ** 2)

        # contributions of the distal caps
        mdc_sum = 0
        Idc_sum = 0
        for j in range(num_rods_per_segment):
            # mass of the distal cap
            mdc = rhoec[i, 0] * ldc[i, 0] * rout[i, j] ** 2
            mdc_sum += mdc

            # moment of inertia of distal cap about its own CoG
            Idc = 1 / 12 * mdc * (3 * rout[i, j] ** 2 + ldc[i, 0] ** 2)
            Idc_sum += Idc

        # contributions of the proximal caps
        mpc_sum = 0
        Ipc_sum = 0
        if i < num_segments - 1:
            for j in range(num_rods_per_segment):
                # mass of the proximal cap
                mpc = rhoec[i + 1, 0] * lpc[i + 1, 0] * rout[i + 1, j] ** 2
                mpc_sum += mpc

                # moment of inertia of distal cap about its own CoG
                Ipc = 1 / 12 * mpc * (3 * rout[i + 1, j] ** 2 + lpc[i + 1, 0] ** 2)
                Ipc_sum += Ipc

        # total mass of the platform including the distal caps
        mp = mp_itself + mdc_sum + mpc_sum

        # position of the CoG with respect to the distal end of the rods
        _contrib_dc_pc = mdc_sum * ldc[i, 0] / 2 + mp_itself * (
            ldc[i, 0] + pcudim[i, 1] / 2
        )
        if i < num_segments - 1:
            relCoGp = (
                sp.Matrix(
                    [
                        0.0,  # TODO: fix in case the rods are not symmetric
                        _contrib_dc_pc
                        + mpc_sum * (ldc[i, 0] + pcudim[i, 1] + lpc[i + 1, 0] / 2),
                    ]
                )
                / mp
            )
        else:
            relCoGp = (
                sp.Matrix(
                    [
                        0.0,  # TODO: fix in case the rods are not symmetric
                        _contrib_dc_pc,
                    ]
                )
                / mp
            )

        # relative offsets
        doffCoGdc = sp.Matrix([0.0, ldc[i, 0] / 2]) - relCoGp
        doffCoGdc_norm = sp.sqrt(doffCoGdc[0] ** 2 + doffCoGdc[1] ** 2)
        doffCoGp = sp.Matrix([0.0, ldc[i, 0] + pcudim[i, 1] / 2]) - relCoGp
        doffCoGp_norm = sp.sqrt(doffCoGp[0] ** 2 + doffCoGp[1] ** 2)
        if i < num_segments - 1:
            doffCoGpc = sp.Matrix([0.0, lpc[i + 1, 0] / 2]) - relCoGp
        else:
            doffCoGpc = sp.Matrix([0.0, 0.0])
        doffCoGpc_norm = sp.sqrt(doffCoGpc[0] ** 2 + doffCoGpc[1] ** 2)

        # total inertia with respect to the CoG
        Ip = sp.simplify(
            Idc_sum
            + Ip_itself
            + Ipc_sum
            + mdc_sum * doffCoGdc_norm**2
            + mp_itself * doffCoGp_norm**2
            + mpc_sum * doffCoGpc_norm**2
        )

        # rotation of the platform
        Rp = R.subs(s, l[i])
        # position of the center of the platform
        pp = p.subs(s, l[i]) + Rp @ sp.Matrix([0.0, ldc[i, 0] + pcudim[i, 1] / 2])
        # pose of the platform
        chip = sp.zeros(3, 1)
        chip[:2, 0] = pp  # the x and y position
        chip[2, 0] = th.subs(s, l[i])  # the orientation angle theta
        chip_sms.append(chip)
        # position of the CoG of the platform
        pCoGp = p.subs(s, l[i]) + Rp @ relCoGp
        # pose of the CoG of the platform
        chiCoGp = sp.zeros(3, 1)
        chiCoGp[:2, 0] = pCoGp  # the x and y position
        chiCoGp[2, 0] = th.subs(s, l[i])  # the orientation angle theta
        # Jacobians of the CoG of the platform
        JCoGp = chiCoGp.jacobian(xi)
        JpCoGp = pCoGp.jacobian(xi)  # positional Jacobian of the platform
        JpCoGo = sp.simplify(sp.Matrix([[chiCoGp[2, 0]]]).jacobian(xi))
        # mass matrix of the platform
        Bp = mp * JpCoGp.T @ JpCoGp + Ip * JpCoGo.T @ JpCoGo
        if simplify:
            Bp = sp.simplify(Bp)
        B = B + Bp
        # potential energy of the platform
        Up = sp.simplify(mp * g.T @ pCoGp)
        U = U + Up

        # update the orientation for the next segment
        th_prev = th.subs(s, l[i])

        # update the position for the next segment
        # add the length of the distal caps and the height of the platform
        p_prev = p.subs(s, l[i]) + Rp @ sp.Matrix([0.0, ldc[i, 0] + pcudim[i, 1]])

    # end-effector pose
    chiee = sp.Matrix([p_prev[0], p_prev[1], th_prev])
    print("chiee =\n", chiee)
    Jee = chiee.jacobian(xi)  # Jacobian of the end-effector
    print("Jee =\n", Jee)
    Jee_d = compute_dAdt(Jee, xi, xi_d)  # time derivative of the end-effector Jacobian
    print("Jee_d =\n", Jee_d)

    # simplify mass matrix
    if simplify:
        B = sp.simplify(B)
    print("B =\n", B)

    C = compute_coriolis_matrix(B, xi, xi_d, simplify=simplify)
    print("C =\n", C)

    # compute the gravity force vector
    G = -U.jacobian(xi).transpose()
    if simplify:
        G = sp.simplify(G)
    print("G =\n", G)

    # dictionary with expressions
    sym_exps = {
        "params_syms": {
            "th0": th0,
            "l": l_syms,
            "lpc": lpc_syms,
            "ldc": ldc_syms,
            "rout": rout_syms,
            "rin": rin_syms,
            "roff": roff_syms,
            "pcudim": pcudim_syms,
            "rhor": rhor_syms,
            "rhop": rhop_syms,
            "rhoec": rhoec_syms,
            "g": g_syms,
        },
        "state_syms": {
            "xi": xi_syms,
            "xi_d": xi_d_syms,
            "s": s,
        },
        "exps": {
            "chiv_sms": chiv_sms,  # list of pose expressions (for the virtual rod along the centerline of each segment)
            # list of pose expressions (for the centerline of each rod).
            # Total length is n_segments * num_rods_per_segment
            "chir_sms": chir_sms,
            "chip_sms": chip_sms,  # expression for the pose of the CoG of the platform of shape (3, )
            "chiee": chiee,  # expression for the pose of the end-effector of shape (3, )
            "Jv_sms": Jv_sms,  # list of the Jacobians of the virtual backbone of each segment
            "Jr_sms": Jr_sms,  # list of the Jacobians of the centerline of each rod
            "Jp_sms": Jp_sms,  # list of the platform Jacobians
            "Jee": Jee,  # Jacobian of the end-effector
            "Jee_d": Jee_d,  # time derivative of the Jacobian of the end-effector
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
