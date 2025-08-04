import dill
from jax import Array, lax
from jax import numpy as jnp
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from .utils import (
    concatenate_params_syms,
    compute_strain_basis,
)

import equinox as eqx

from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
    Tsit5,
    PIDController,
    ConstantStepSize,
    AbstractSolver,
)

class PlanarHSA(eqx.Module):
    """
    TODO: Add docstring for PlanarHSA class.

    Args:
        eqx (_type_): _description_
    """
    
    global_eps: float = 1e-6
    
    consider_hysteresis: bool = eqx.static_field()
    num_hysteresis: int = eqx.static_field()
    B_hyst: Array
    hyst_alpha:Array
    hyst_A: Array
    hyst_n: Array
    hyst_beta: Array
    hyst_gamma: Array
    
    num_segments        : int = eqx.static_field()
    num_rods_per_segment: int = eqx.static_field()
    num_dofs            : int = eqx.static_field()

    B_xi: Array
    
    params_syms         : Dict[str, Array]  #= eqx.static_field()
    params_for_lambdify : List[Array]       #= eqx.static_field()
    
    L       : Array
    L_cum   : Array
    Lmax    : Array # Maximum length of the robot (sum of all segments)
    
    chiv_lambda_sms: List[Callable]
    chir_lambda_sms: List[Callable]
    chip_lambda_sms: List[Callable]
    
    chiee_lambda: Callable
    Jee_lambda: Callable
    Jeed_lambda: Callable
    
    B_lambda: Callable
    C_lambda: Callable
    G_lambda: Callable
    Shat_lambda: Callable
    K_lambda: Callable
    D_lambda: Callable
    alpha_lambda: Callable
    
    roff: Array
    kappa_b_eq: Array
    sigma_sh_eq: Array
    sigma_a_eq: Array
    
    pcudim: Array
    lpc: Array
    ldc: Array
    chiee_off: Array
    
    def __init__(
        self,
        sym_exp_filepath: Union[str, Path],
        params: Dict[str, Array] = None,
        strain_selector: Array = None,
        global_eps: float = 1e-6,
        consider_hysteresis: bool = False,
    )-> 'PlanarHSA':
        """
        Initialize the PlanarHSA system.

        Args:
            sym_exp_filepath: path to file containing symbolic expressions
            strain_selector: array of shape (num_dofs, ) with boolean values indicating which components of the
                    strain are active / non-zero
            global_eps: small number to avoid singularities (e.g., division by zero)
            consider_hysteresis: If True, Bouc-Wen is used to model hysteresis. Otherwise, hysteresis will be neglected.
        """
        self.global_eps = global_eps
    
        # Load saved symbolic data
        try:
            sym_exps = dill.load(open(str(sym_exp_filepath), "rb"))
        except FileNotFoundError:
            return FileNotFoundError(
                f"Symbolic expressions file {sym_exp_filepath} not found. Please generate the symbolic expressions first."
            )
        
        # Parameters numerical values
        try:
            roff = params["roff"]
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'roff'. Please generate the symbolic expressions first."
            )
        self.roff = roff
        try:
            kappa_b_eq = params["kappa_b_eq"]
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'kappa_b_eq'. Please generate the symbolic expressions first."
            )
        self.kappa_b_eq = kappa_b_eq
        try:
            sigma_sh_eq = params["sigma_sh_eq"]
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'sigma_sh_eq'. Please generate the symbolic expressions first."
            )
        self.sigma_sh_eq = sigma_sh_eq
        try:
            sigma_a_eq = params["sigma_a_eq"]
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'sigma_a_eq'. Please generate the symbolic expressions first."
            )
        self.sigma_a_eq = sigma_a_eq
        
        try:
            pcudim = params["pcudim"]
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'pcudim'. Please generate the symbolic expressions first."
            )
        self.pcudim = pcudim
        try:
            lpc = params["lpc"]
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'lpc'. Please generate the symbolic expressions first."
            )
        self.lpc = lpc
        try:
            ldc = params["ldc"]
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'ldc'. Please generate the symbolic expressions first."
            )
        self.ldc = ldc
        try:
            chiee_off = params["chiee_off"]
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'chiee_off'. Please generate the symbolic expressions first."
            )
        self.chiee_off = chiee_off
        
        # Symbols for robot parameters
        try:
            params_syms = sym_exps["params_syms"]
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'params_syms'. Please generate the symbolic expressions first."
            )
        self.params_syms = params_syms
        
        try:
            params_for_lambdify = []
            for params_key, params_vals in sorted(params.items()):
                if params_key in self.params_syms.keys():
                    for param in params_vals.flatten():
                        params_for_lambdify.append(param)
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain the required parameters. Please generate the symbolic expressions first."
            )
        self.params_for_lambdify = params_for_lambdify
        
        try: 
            L = params["l"]
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'l'. Please generate the symbolic expressions first."
            )
        self.L = L
        
        try: 
            # cumsum of the segment lengths
            L_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), self.L]))
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'l'. Please generate the symbolic expressions first."
            )
        self.L_cum = L_cum
        
        # Maximum length of the robot
        self.Lmax = L_cum[-1]
        
        # Number of segments
        try:
            num_segments = len(params_syms["l"])
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'l'. Please generate the symbolic expressions first."
            )
        self.num_segments = num_segments
        
        # Number of rods per segment
        try:
            num_rods_per_segment = len(params_syms["rout"]) // num_segments
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'rout'. Please generate the symbolic expressions first."
            )
        self.num_rods_per_segment = num_rods_per_segment

        # =================================================
        # Parameters
        # =====================

        # concatenate the robot params symbols
        params_syms_cat = concatenate_params_syms(params_syms)

        # Number of degrees of freedom
        try:
            num_dofs = len(sym_exps["state_syms"]["xi"])
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'state_syms'. Please generate the symbolic expressions first."
            )
        self.num_dofs = num_dofs
        
        # Hysteresis
        self.consider_hysteresis = consider_hysteresis
        
        if consider_hysteresis:
            try: 
                hyst_params = params["hysteresis"]
            except KeyError:
                raise KeyError(
                    f"Symbolic expressions file {sym_exp_filepath} does not contain 'hysteresis' parameters. Please generate the symbolic expressions first."
                )
            try:
                B_hyst = hyst_params["basis"]
            except KeyError:
                raise KeyError(
                    f"Symbolic expressions file {sym_exp_filepath} does not contain 'hysteresis' basis. Please generate the symbolic expressions first."
                )
            self.B_hyst = B_hyst
            
            try:
                num_hysteresis = B_hyst.shape[1]
            except AttributeError:
                raise AttributeError(
                    f"Symbolic expressions file {sym_exp_filepath} does not contain 'hysteresis' basis. Please generate the symbolic expressions first."
                )
            self.num_hysteresis = num_hysteresis
            
            try:
                hyst_alpha = params["hysteresis"]["alpha"]
            except KeyError:
                raise KeyError(
                    f"Symbolic expressions file {sym_exp_filepath} does not contain 'hysteresis' alpha. Please generate the symbolic expressions first."
                )
            self.hyst_alpha = hyst_alpha
            
            try:
                hyst_A = hyst_params["A"]
            except KeyError:
                raise KeyError(
                    f"Symbolic expressions file {sym_exp_filepath} does not contain 'hysteresis' A. Please generate the symbolic expressions first."
                )
            self.hyst_A = hyst_A
            
            try:
                hyst_n = hyst_params["n"]
            except KeyError:
                raise KeyError(
                    f"Symbolic expressions file {sym_exp_filepath} does not contain 'hysteresis' n. Please generate the symbolic expressions first."
                )
            self.hyst_n = hyst_n
            
            try:
                hyst_beta = hyst_params["beta"]
            except KeyError:
                raise KeyError(
                    f"Symbolic expressions file {sym_exp_filepath} does not contain 'hysteresis' beta. Please generate the symbolic expressions first."
                )
            self.hyst_beta = hyst_beta
            
            try: 
                hyst_gamma = hyst_params["gamma"]
            except KeyError:
                raise KeyError(
                    f"Symbolic expressions file {sym_exp_filepath} does not contain 'hysteresis' gamma. Please generate the symbolic expressions first."
                )
            self.hyst_gamma = hyst_gamma
        else:
            self.num_hysteresis = 0
            self.B_hyst = jnp.zeros((num_dofs, 0))
            self.hyst_alpha = jnp.zeros((num_dofs,))
            self.hyst_A = jnp.zeros((1,))
            self.hyst_n = jnp.zeros((1,))
            self.hyst_beta = jnp.zeros((1,))
            self.hyst_gamma = jnp.zeros((1,))
        

        # compute the strain basis
        if strain_selector is None:
            strain_selector = jnp.ones((num_dofs,), dtype=bool)
        else:
            if not isinstance(strain_selector, (list, jnp.ndarray)):
                raise TypeError(
                    f"strain_selector must be a list or an array, got {type(strain_selector).__name__}"
                )
            strain_selector = jnp.asarray(strain_selector)
            if not jnp.issubdtype(strain_selector.dtype, jnp.bool_):
                raise TypeError(
                    f"strain_selector must be a boolean array, got {strain_selector.dtype}"
                )
            if strain_selector.size != num_dofs:
                raise ValueError(
                    f"strain_selector must have {num_dofs} elements, got {strain_selector.size}"
                )
            strain_selector = strain_selector.reshape(num_dofs)
        self.B_xi = compute_strain_basis(strain_selector)

        # concatenate the list of state symbols
        try: 
            state_syms_cat = sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["xid"]
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain 'state_syms'. Please generate the symbolic expressions first."
            )

        # =================================================
        # lambdify symbolic expressions
        
        chiv_lambda_sms = []
        # iterate through symbolic expressions for each segment
        try:
            for chiv_exp in sym_exps["exps"]["chiv_sms"]:
                chiv_lambda = sp.lambdify(
                    params_syms_cat
                    + sym_exps["state_syms"]["xi"]
                    + [sym_exps["state_syms"]["s"]],
                    chiv_exp,
                    "jax",
                )
                chiv_lambda_sms.append(chiv_lambda)
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does ['exps']['chiv_sms']. Please generate the symbolic expressions first."
            )
        self.chiv_lambda_sms = chiv_lambda_sms

        chir_lambda_sms = []
        # iterate through symbolic expressions for each segment
        try:
            for chir_exp in sym_exps["exps"]["chir_sms"]:
                chir_lambda = sp.lambdify(
                    params_syms_cat
                    + sym_exps["state_syms"]["xi"]
                    + [sym_exps["state_syms"]["s"]],
                    chir_exp,
                    "jax",
                )
                chir_lambda_sms.append(chir_lambda)
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain ['exps']['chir_sms']. Please generate the symbolic expressions first."
            )
        self.chir_lambda_sms = chir_lambda_sms

        chip_lambda_sms = []
        # iterate through symbolic expressions for each segment
        try:
            for chip_exp in sym_exps["exps"]["chip_sms"]:
                chip_lambda = sp.lambdify(
                    params_syms_cat + sym_exps["state_syms"]["xi"],
                    chip_exp,
                    "jax",
                )
                chip_lambda_sms.append(chip_lambda)
        except KeyError:
            return KeyError(
                f"Symbolic expressions file {sym_exp_filepath} does not contain ['exps']['chip_sms']. Please generate the symbolic expressions first."
            )
        self.chip_lambda_sms = chip_lambda_sms

        # end-effector kinematics
        try:
            chiee_lambda = sp.lambdify(
                params_syms_cat + sym_exps["state_syms"]["xi"],
                sym_exps["exps"]["chiee"],
                "jax",
            )
        except ValueError:
            return "Fail to lambdify chiee. Check the symbolic expressions file."
        self.chiee_lambda = chiee_lambda
        
        try:    
            Jee_lambda = sp.lambdify(
                params_syms_cat + sym_exps["state_syms"]["xi"],
                sym_exps["exps"]["Jee"],
                "jax",
            )
        except ValueError:
            return "Fail to lambdify Jee. Check the symbolic expressions file."
        self.Jee_lambda = Jee_lambda
        
        try:
            Jeed_lambda = sp.lambdify(
                params_syms_cat + sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["xid"],
                sym_exps["exps"]["Jeed"],
                "jax",
            )
        except ValueError:
            return "Fail to lambdify Jeed. Check the symbolic expressions file."
        self.Jeed_lambda = Jeed_lambda

        # dynamical matrices
        try:
            B_lambda = sp.lambdify(
                params_syms_cat + sym_exps["state_syms"]["xi"], sym_exps["exps"]["B"], "jax"
            )
        except ValueError:
            return "Fail to lambdify B. Check the symbolic expressions file."
        self.B_lambda = B_lambda
        
        try:
            C_lambda = sp.lambdify(
                params_syms_cat + state_syms_cat, sym_exps["exps"]["C"], "jax"
            )
        except ValueError:
            return "Fail to lambdify C. Check the symbolic expressions file."
        self.C_lambda = C_lambda
        
        try:
            G_lambda = sp.lambdify(
                params_syms_cat + sym_exps["state_syms"]["xi"], sym_exps["exps"]["G"], "jax"
            )
        except ValueError:
            return "Fail to lambdify G. Check the symbolic expressions file."
        self.G_lambda = G_lambda
        
        try:
            Shat_lambda = sp.lambdify(params_syms_cat, sym_exps["exps"]["Shat"], "jax")
        except ValueError:
            return "Fail to lambdify Shat. Check the symbolic expressions file."
        self.Shat_lambda = Shat_lambda
        
        try:
            K_lambda = sp.lambdify(
                params_syms_cat + sym_exps["state_syms"]["xi"], sym_exps["exps"]["K"], "jax"
            )
        except ValueError:
            return "Fail to lambdify K. Check the symbolic expressions file."
        self.K_lambda = K_lambda
        
        try: 
            D_lambda = sp.lambdify(
                params_syms_cat, sym_exps["exps"]["D"], "jax"
            )
        except ValueError:
            return "Fail to lambdify D. Check the symbolic expressions file."
        self.D_lambda = D_lambda
        
        try: 
            alpha_lambda = sp.lambdify(
                params_syms_cat + sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["phi"],
                sym_exps["exps"]["alpha"],
                "jax",
            )
        except ValueError:
            return "Fail to lambdify alpha. Check the symbolic expressions file."
        self.alpha_lambda = alpha_lambda

    def beta_fn(
        self,
        vxi: Array) -> Array:
        """
        Map the generalized coordinates to the strains in the physical rods
        Args:
            vxi: strains of the virtual backbone of shape (num_dofs, )
        Returns:
            pxi: strains in the physical rods of shape (num_segments, num_rods_per_segment, 3)
        """
        # strains of the virtual rod
        vxi = vxi.reshape((self.num_segments, 1, -1))

        pxi = jnp.repeat(vxi, self.num_rods_per_segment, axis=1)
        psigma_a = (
            pxi[:, :, 2]
            + self.roff * jnp.repeat(vxi, self.num_rods_per_segment, axis=1)[..., 0]
        )
        pxi = pxi.at[:, :, 2].set(psigma_a)

        return pxi

    def beta_inv_fn(
        self,
        pxi: Array) -> Array:
        """
        Map the strains in the physical rods to the strains of the virtual backbone
        Args:
            pxi: strains in the physical rods of shape (num_segments, num_rods_per_segment, 3)
        Returns:
            vxi: strains of the virtual backbone of shape (num_dofs, )
        """
        vxi = jnp.mean(pxi, axis=1)
        vxi = vxi.at[:, 2].set(
            vxi[:, 2] - jnp.mean(self.roff * pxi[..., 0], axis=1)
        )
        vxi = vxi.flatten()

        return vxi

    def rest_strains_fn(
        self,
    ) -> Array:
        """
        Compute the rest strains of the virtual backbone

        Returns:
            vxi_star: rest strains of the virtual backbone of shape (num_dofs, )
        """
        # rest strains of the physical rods
        pxi_star = jnp.zeros((self.num_segments, self.num_rods_per_segment, 3))
        pxi_star = pxi_star.at[:, :, 0].set(self.kappa_b_eq)
        pxi_star = pxi_star.at[:, :, 1].set(self.sigma_sh_eq)
        pxi_star = pxi_star.at[:, :, 2].set(self.sigma_a_eq)

        # map the rest strains from the physical rods to the virtual backbone
        vxi_star = self.beta_inv_fn(pxi_star)
        return vxi_star
    
    def classify_segment(
        self,
        s: Array,
    ) -> Tuple[Array, Array]:
        """
        Classify the point along the robot to the corresponding segment.

        Args:
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            segment_idx (Array): index of the segment where the point is located
            s_local (Array): point coordinate along the segment in the interval [0, l_segment]
        """

        # Classify the point along the robot to the corresponding segment
        segment_idx = jnp.clip(jnp.sum(s > self.L_cum) - 1, 0, self.num_segments - 1)

        # Compute the point coordinate along the segment in the interval [0, l_segment]
        s_local = s - self.L_cum[segment_idx]

        return segment_idx, s_local

    def strain(
        self, 
        q: Array
    ) -> Array:
        """
        Map the generalized coordinates to the strains in the virtual backbone
        Args:
            q: generalized coordinates of shape (num_dofs, )
        Returns:
            xi: strains of the virtual backbone of shape (num_dofs, )
        """
        # rest strains of the virtual backbone
        xi_star = self.rest_strains_fn()

        # map the configuration to the strains
        xi = self.B_xi @ q + xi_star

        return xi

    def apply_eps_to_bend_strains_fn(
        self,
        xi: Array, 
        eps: Optional[float] = global_eps
        ) -> Array:
        """
        Add a small number to the bending strain to avoid singularities
        """
        xi_reshaped = xi.reshape((-1, 3))

        xi_bend_sign = jnp.sign(xi_reshaped[:, 0])
        # set zero sign to 1 (i.e. positive)
        xi_bend_sign = jnp.where(xi_bend_sign == 0, 1, xi_bend_sign)
        # add eps to the bending strain (i.e. the first column)
        sigma_b_epsed = lax.select(
            jnp.abs(xi_reshaped[:, 0]) < eps,
            xi_bend_sign * eps,
            xi_reshaped[:, 0],
        )
        xi_epsed = jnp.stack(
            [
                sigma_b_epsed,
                xi_reshaped[:, 1],
                xi_reshaped[:, 2],
            ],
            axis=1,
        )

        # flatten the array
        xi_epsed = xi_epsed.flatten()

        return xi_epsed

    def forward_kinematics_virtual_backbone_fn(
        self,
        q: Array, 
        s: Array, 
    ) -> Array:
        """
        Evaluate the forward kinematics the virtual backbone
        Args:
            q: generalized coordinates of shape (num_dofs, )
            s: point coordinate along the rod in the interval [0, L].
        Returns:
            chi: pose of the backbone point in Cartesian-space with shape (3, )
                Consists of [theta, p_x, p_y]
                where theta is the planar orientation with respect to the x-axis,
                p_x is the x-position, p_y is the y-position,
        """
        # map the configuration to the strains
        xi = self.strain(q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains_fn(xi)

        # determine in which segment the point is located
        segment_idx, s_local = self.classify_segment(s)

        chi = lax.switch(
            segment_idx, self.chiv_lambda_sms, *self.params_for_lambdify, *xi_epsed, s_local
        ).squeeze()
        
        chi = jnp.roll(chi, 1)

        return chi

    def forward_kinematics_rod_fn(
        self,
        q: Array,
        s: Array,
        rod_idx: Array,
    ) -> Array:
        """
        Evaluate the forward kinematics of the physical rods
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (num_dofs, )
            s: point coordinate along the rod in the interval [0, L].
            rod_idx: index of the rod. If there are two rods per segment, then rod_idx can be 0 or 1.
        Returns:
            chir: pose of the rod centerline point in Cartesian-space with shape (3, )
                Consists of [theta, p_x, p_y]
                where theta is the planar orientation with respect to the x-axis,
                p_x is the x-position, p_y is the y-position,
        """
        # map the configuration to the strains
        xi = self.strain(q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains_fn(xi)

        # determine in which segment the point is located
        segment_idx, s_local = self.classify_segment(s)

        chir_lambda_sms_idx = segment_idx * self.num_rods_per_segment + rod_idx
        chir = lax.switch(
            chir_lambda_sms_idx,
            self.chir_lambda_sms,
            *self.params_for_lambdify,
            *xi_epsed,
            s_local,
        ).squeeze()
        
        chir = jnp.roll(chir, 1)

        return chir

    def forward_kinematics_platform_fn(
        self,
        q: Array, 
        segment_idx: Array
    ) -> Array:
        """
        Evaluate the forward kinematics the platform
        Args:
            q: generalized coordinates of shape (num_dofs, )
            segment_idx: index of the segment
        Returns:
            chip: pose of the CoG of the platform in Cartesian-space with shape (3, )
                Consists of [theta, p_x, p_y]
                where theta is the planar orientation with respect to the x-axis,
                p_x is the x-position, p_y is the y-position,
        """
        # map the configuration to the strains
        xi = self.strain(q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains_fn(xi)

        chip = lax.switch(
            segment_idx, 
            self.chip_lambda_sms, 
            *self.params_for_lambdify, 
            *xi_epsed
        ).squeeze()
        
        chip = jnp.roll(chip, 1)

        return chip

    def forward_kinematics_end_effector_fn(
        self,
        q: Array
    ) -> Array:
        """
        Evaluate the forward kinematics of the end-effector
        Args:
            q: generalized coordinates of shape (num_dofs, )
        Returns:
            chiee: pose of the end-effector in Cartesian-space of shape (3, )
                Consists of [theta, p_x, p_y]
                where theta is the planar orientation with respect to the x-axis,
                p_x is the x-position, p_y is the y-position,
        """
        # map the configuration to the strains
        xi = self.strain(q)
        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains_fn(xi)

        # evaluate the symbolic expression
        chiee = self.chiee_lambda(
            *self.params_for_lambdify, 
            *xi_epsed
        ).squeeze()
        
        chiee = jnp.roll(chiee, 1)

        return chiee

    def jacobian_end_effector_fn(
        self,
        q: Array
    ) -> Array:
        """
        Evaluate the Jacobian of the end-effector
        Args:
            q: generalized coordinates of shape (num_dofs, )
        Returns:
            Jee: the Jacobian of the end-effector pose with respect to the generalized coordinates.
                Jee is an array of shape (3, num_dofs).
        """
        # map the configuration to the strains
        xi = self.strain(q)
        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains_fn(xi)

        # evaluate the symbolic expression
        Jee = self.Jee_lambda(
            *self.params_for_lambdify, 
            *xi_epsed
        )

        return Jee

    def inverse_kinematics_end_effector_fn(
        self,
        chiee: Array
    ) -> Array:
        """
        Evaluates the inverse kinematics for a given end-effector pose.
            Important: only works for one segment!
        Args:
            params: Dictionary of robot parameters
            chiee: pose of the end-effector in Cartesian-space of shape (3, )
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            q: generalized coordinates of shape (num_dofs, )
        """
        assert self.num_segments == 1, "Inverse kinematics only works for one segment!"

        # height of platform
        hp = self.pcudim[0, 1]
        # length of the proximal rod caps
        lpc = self.lpc[0]
        # length of the distal rod caps
        ldc = self.ldc[0]
        # offset of the end-effector from the distal surface of the platform
        chiee_off = self.chiee_off

        # transformation from the base to the proximal end of the virtual backbone
        T_b_to_pe = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, lpc],
                [0.0, 0.0, 1.0],
            ]
        )

        # transformation from the base to the end-effector
        T_b_to_ee = jnp.array(
            [
                [jnp.cos(chiee[2]), -jnp.sin(chiee[2]), chiee[0]],
                [jnp.sin(chiee[2]), jnp.cos(chiee[2]), chiee[1]],
                [0.0, 0.0, 1.0],
            ]
        )

        # transformation from the distal end of the virtual backbone to the end-effector
        T_de_to_ee = jnp.array(
            [
                [jnp.cos(chiee_off[2]), -jnp.sin(chiee_off[2]), chiee_off[0]],
                [jnp.sin(chiee_off[2]), jnp.cos(chiee_off[2]), ldc + hp + chiee_off[1]],
                [0.0, 0.0, 1.0],
            ]
        )

        # compute the transformation from the proximal to the distal end of the virtual backbone
        T_pe_to_de = jnp.linalg.inv(T_b_to_pe) @ T_b_to_ee @ jnp.linalg.inv(T_de_to_ee)

        # compute the SE(2) pose from the transformation matrix
        vchi_pe_to_de = jnp.array(
            [
                T_pe_to_de[0, 2],
                T_pe_to_de[1, 2],
                jnp.arctan2(T_pe_to_de[1, 0], T_pe_to_de[0, 0]),
            ]
        )

        # extract the x and y position and the orientation
        px, py, th = vchi_pe_to_de[0], vchi_pe_to_de[1], vchi_pe_to_de[2]

        # add small eps for numerical stability
        th_sign = jnp.sign(th)
        # set zero sign to 1 (i.e. positive)
        th_sign = jnp.where(th_sign == 0, 1, th_sign)
        # add eps to the bending strain (i.e. the first column)
        th_epsed = th + th_sign * self.global_eps

        # compute the inverse kinematics for the virtual backbone
        vxi = (
            th_epsed
            / (2 * self.Lmax)
            * jnp.array(
                [
                    2,
                    py - (px * jnp.sin(th_epsed)) / (jnp.cos(th_epsed) - 1),
                    -px - (py * jnp.sin(th_epsed)) / (jnp.cos(th_epsed) - 1),
                ]
            )
        )

        # rest strains of the virtual backbone
        vxi_star = self.rest_strains_fn()

        # map the strains to the generalized coordinates
        q = jnp.linalg.pinv(self.B_xi) @ (vxi - vxi_star)

        return q

    def _inertia_full_matrix(
        self,
        q: Array,
        eps: float = 1e4 * global_eps
    ) -> Array:
        """
        Compute the full inertia matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            eps (float): small number to avoid singularities (e.g., division by zero).

        Returns:
            B_full (Array): Full inertia matrix of shape (num_dofs_max, num_dofs_max).
        """
        xi = self.strain(q)
        
        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains_fn(xi, eps)
        
        B_full = self.B_lambda(*self.params_for_lambdify, *xi_epsed)

        return B_full

    def inertia_matrix(
        self,
        q: Array,
        eps: float = 1e4 * global_eps
    ) -> Array:
        """
        Compute the inertia matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            eps (float): small number to avoid singularities (e.g., division by zero).

        Returns:
            B (Array): Inertia matrix of shape (num_dofs, num_dofs).
        """
        B_full = self._inertia_full_matrix(q, eps)

        B = self.B_xi.T @ B_full @ self.B_xi

        return B

    def _coriolis_full_matrix(
        self,
        q: Array,
        qd: Array,
        eps: float = 1e4 * global_eps
    ) -> Array:
        """
        Compute the full Coriolis matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_dofs,).
            eps (float): small number to avoid singularities (e.g., division by zero).

        Returns:
            C_full (Array): Full Coriolis matrix of shape (num_dofs_max, num_dofs_max).
        """
        xi = self.strain(q)
        xid = self.B_xi @ qd
        
        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains_fn(xi, eps)
        
        C_full = self.C_lambda(
            *self.params_for_lambdify, 
            *xi_epsed, 
            *xid
        )

        return C_full

    def coriolis_matrix(
        self,
        q: Array,
        qd: Array,
        eps: float = 1e4 * global_eps
    ) -> Array:
        """
        Compute the Coriolis matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_dofs,).
            eps (float): small number to avoid singularities (e.g., division by zero).

        Returns:
            C (Array): Coriolis matrix of shape (num_dofs, num_dofs).
        """
        C_full = self._coriolis_full_matrix(q, qd, eps)

        C = self.B_xi.T @ C_full @ self.B_xi

        return C

    def _gravitational_full_vector(
        self,
        q: Array,
        eps: float = 1e4 * global_eps
    ) -> Array:
        """
        Compute the full gravitational vector of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            eps (float): small number to avoid singularities (e.g., division by zero).

        Returns:
            G (Array): Full gravitational vector of shape (num_dofs_max,).
        """

        xi = self.strain(q)
        
        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains_fn(xi, eps)
        
        G_full = self.G_lambda(*self.params_for_lambdify, *xi_epsed).squeeze()

        return G_full

    def gravitational_vector(
        self,
        q: Array,
        eps: float = 1e4 * global_eps
    ) -> Array:
        """
        Compute the gravitational vector of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            eps (float): small number to avoid singularities (e.g., division by zero).

        Returns:
            G (Array): Gravitational vector of shape (num_dofs,).
        """
        G_full = self._gravitational_full_vector(q, eps)

        G = self.B_xi.T @ G_full

        return G

    def _stiffness_full_vector(
        self,
        q: Array,
        eps: float = 1e4 * global_eps
    ) -> Array:
        """
        Compute the full stiffness vector of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            eps (float): small number to avoid singularities (e.g., division by zero).
        
        Returns:
            K_full (Array): Full stiffness vector of shape (num_dofs_max, ).
        """
        xi = self.strain(q)
        
        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains_fn(xi, eps)
        
        K_full = self.K_lambda(*self.params_for_lambdify, *xi_epsed).squeeze()

        return K_full

    def stiffness_vector(
        self,
        q: Array,
        eps: float = 1e4 * global_eps
    ) -> Array:
        """
        Compute the stiffness vector of the robot.
        
        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            eps (float): small number to avoid singularities (e.g., division by zero).

        Returns:
            K (Array): Stiffness vector of shape (num_dofs, ).
        """
        K_full = self._stiffness_full_vector(q, eps)
        
        K = self.B_xi.T @ K_full

        return K

    def _damping_full_matrix(
        self,
    ) -> Array:
        """
        Compute the full damping matrix of the robot.

        Args:
            None

        Returns:
            D (Array): Full damping matrix of shape (num_dofs_max, num_dofs_max).
        """
        D_full = self.D_lambda(*self.params_for_lambdify)

        return D_full

    def damping_matrix(
        self,
    ) -> Array:
        """
        Compute the damping matrix of the robot.

        Args:
            None

        Returns:
            D (Array): Damping matrix of shape (num_dofs, num_dofs).
        """
        D_full = self._damping_full_matrix()

        D = self.B_xi.T @ D_full @ self.B_xi

        return D

    def _actuation_full_mapping(
        self,
        q: Array,
        phi: Array
    ) -> Array:
        """
        Compute the actuation mapping of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            phi (Array): motor positions / twist angles of shape (num_segments * num_rods_per_segment, )

        Returns:
            alpha (Array): Actuation mapping of shape (num_dofs, num_dofs).
        """
        xi = self.strain(q)
        
        alpha = self.alpha_lambda(
            *self.params_for_lambdify, 
            *xi,
            *phi
        ).squeeze()

        return alpha
    
    def actuation_mapping(
        self,
        q: Array,
        phi: Array
    ) -> Array:
        """
        Compute the actuation mapping of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            phi (Array): motor positions / twist angles of shape (num_segments * num_rods_per_segment, )

        Returns:
            alpha (Array): Actuation mapping of shape (num_dofs, num_dofs).
        """
        alpha = self._actuation_full_mapping(q, phi)

        # apply the strain basis
        alpha = self.B_xi.T @ alpha

        return alpha
    
    def Shat(
        self
    )->Array:
        """
        TODO

        Returns:
            Array: TODO
        """
        Shat    = self.Shat_lambda(*self.params_for_lambdify)
        
        return Shat

    def dynamical_matrices(
        self,
        q: Array,
        qd: Array,
        z: Optional[Array] = None,
        phi: Optional[Array] = None,
        eps: float = 1e4 * global_eps,
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute the dynamical matrices of the system.
        Args:
            q: generalized coordinates of shape (num_dofs, )
            qd: generalized velocities of shape (num_dofs, )
            z: state variables of the hysteresis model of shape (n_z, )
            phi: motor positions / twist angles of shape (num_segments * num_rods_per_segment, )
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            B: mass / inertia matrix of shape (num_dofs, num_dofs)
            C: coriolis / centrifugal matrix of shape (num_dofs, num_dofs)
            G: gravity vector of shape (num_dofs, )
            K: elastic vector of shape (num_dofs, )
            D: dissipative matrix of shape (num_dofs, num_dofs)
            alpha: actuation vector acting on the generalized coordinates of shape (num_dofs, )
        """
        if phi is None:
            phi = jnp.zeros((self.num_segments * self.num_rods_per_segment,))
        B = self.inertia_matrix(q, eps)
        C = self.coriolis_matrix(q, qd, eps)
        G = self.gravitational_vector(q, eps)
        K = self.stiffness_vector(q, eps)
        D = self.damping_matrix()
        alpha = self.actuation_mapping(q, phi)
        
        if self.consider_hysteresis is True:
            Shat = self.Shat()
            # add the post-yield potential forces
            K = self.hyst_alpha * K + (1 - self.hyst_alpha) * Shat @ (self.B_hyst @ z)

            # TODO: add post-yield potential forces (i.e., hysteresis effects) to the actuation vector

        return B, C, G, K, D, alpha

    def operational_space_dynamical_matrices(
        self,
        q: Array,
        qd: Array,
        eps: float = 1e4 * global_eps,
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """
        Compute the dynamics in operational space.
        The implementation is based on Chapter 7.8 of "Modelling, Planning and Control of Robotics" by
        Siciliano, Sciavicco, Villani, Oriolo.
        
        Args:
            q: generalized coordinates of shape (num_dofs,)
            qd: generalized velocities of shape (num_dofs,)
            eps: small number to avoid singularities (e.g., division by zero)
        
        Returns:
            Lambda: inertia matrix in the operational space of shape (n_x, n_x)
            mu: matrix with corioli and centrifugal terms in the operational space of shape (n_x, n_x)
            Jee: Jacobian of the end-effector pose with respect to the generalized coordinates of shape (3, num_dofs)
            Jeed: time-derivative of the Jacobian of the end-effector pose with respect to the generalized coordinates
            JeeB_pinv: Dynamically-consistent pseudo-inverse of the Jacobian. Allows the mapping of torques
                from the generalized coordinates to the operational space: f = JB_pinv.T @ tau_q
                Shape (num_dofs, n_x)
        """
        # map the configuration to the strains
        xi = self.strain(q)
        xid = self.B_xi @ qd
        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains_fn(xi, eps)

        # end-effector Jacobian and its time-derivative
        Jee = self.Jee_lambda(*self.params_for_lambdify, *xi_epsed)
        Jeed = self.Jeed_lambda(*self.params_for_lambdify, *xi_epsed, *xid)

        # inverse of the inertia matrix in the configuration space
        B = self.inertia_matrix(q, eps)
        B_inv = jnp.linalg.inv(B)
        
        C = self.coriolis_matrix(q, qd, eps)

        Lambda = jnp.linalg.inv(
            Jee @ B_inv @ Jee.T
        )  # inertia matrix in the operational space
        mu = Lambda @ (
            Jee @ B_inv @ C - Jeed
        )  # coriolis and centrifugal matrix in the operational space

        JeeB_pinv = (
            B_inv @ Jee.T @ Lambda
        )  # dynamically-consistent pseudo-inverse of the Jacobian

        return Lambda, mu, Jee, Jeed, JeeB_pinv
    
    @eqx.filter_jit
    def forward_dynamics(
        self,
        t: float,
        y: Array,
        actuation_args: Tuple[Array, Callable, bool] = None,
    ) -> Array:
        """
        Forward dynamics function.

        Args:
            t (float): Current time.
            y (Array): State vector containing configuration, velocity, and possibly hysteresis state.
                Shape is (2 * num_dofs + num_hysteresis,).
            actuation_args (Tuple): Additional arguments for the actuation function.
                - u (Array): Initial actuation input.
                    If consider_underactuation_model is True, this is an array of shape (num_hysteresis, ) with
                    motor positions / twist angles of the proximal end of the rods.
                    If consider_underactuation_model is False, this is an array of shape (num_dofs, ) with
                    the configuration-space torques.
                - control_fn (Callable): Callable that returns the forcing function of the form control_fn(t, x) -> phi. If consider_underactuation_model is True,
                    then phi is an array of shape (num_dofs, ) with the configuration-space torques. If consider_underactuation_model is False,
                    then phi is an array of shape (num_hysteresis, ) with the motor positions / twist angles of the proximal end of the rods.
                - consider_underactuation_model (bool): If True, the underactuation model is considered. Otherwise, the fully-actuated
                    model is considered with the identity matrix as the actuation matrix.
                
        Returns:
            y_d: Time derivative of the state vector of shape (2 * num_dofs + num_hysteresis, ).
        """
        u, control_fn, consider_underactuation_model = actuation_args
        
        q, qd, z = jnp.split(
            y, [self.num_dofs, 2*self.num_dofs]
        )
              
        zd = (self.B_hyst.T @ qd) * (
            self.hyst_A
            - jnp.abs(z) ** self.hyst_n
            * (
                self.hyst_gamma
                + self.hyst_beta * jnp.sign((self.B_hyst.T @ qd) * z)
            )
        )
        
        if control_fn is not None:
            u = u + control_fn(t, y)
            
        if consider_underactuation_model is True: 
            phi = u       
            B, C, G, K, D, alpha = self.dynamical_matrices(
                q,
                qd,
                z=z,
                phi=phi,
            )
        else:
            B, C, G, K, D, _ = self.dynamical_matrices(
                q,
                qd,
                z=z,
                phi=jnp.zeros((self.num_segments * self.num_rods_per_segment,)),
            )
            alpha = u

        # Inverse of the inertia matrix
        B_inv = jnp.linalg.inv(B)
        
        # Compute the acceleration
        qdd = B_inv @ (-C @ qd - G - K - D @ qd + alpha)

        yd = jnp.concatenate([qd, qdd, zd])

        return yd

    def resolve_upon_time(
        self,
        q0: Array,
        qd0: Array,
        u0: Array,
        control_fn: Optional[Callable] = None,
        consider_underactuation_model: Optional[bool] = True,
        t0: Optional[float] = 0.0,
        t1: Optional[float] = 10.0,
        dt: Optional[float] = 1e-4,
        skip_steps: Optional[int] = 0,
        solver: Optional[AbstractSolver] = Tsit5(),
        stepsize_controller: Optional[PIDController] = ConstantStepSize(),
        max_steps: Optional[int] = None,
    ) -> Tuple[Array, Array, Array]:
        """
        Resolve the system dynamics over time using Diffrax.

        Args:
            q0 (Array): Initial configuration (strains).
            qd0 (Array): Initial velocity (strains).
            u0 (Array): Initial actuation input.
                If consider_underactuation_model is True, 
                    array of shape (num_hysteresis, ) with
                    motor positions / twist angles of the proximal end of the rods.
                If consider_underactuation_model is False, 
                    array of shape (num_dofs, ) with
                    the configuration-space torques.
            control_fn (Callable, optional): Callable that returns the forcing function of the form control_fn(t, [q, qd]) -> phi. 
                If consider_underactuation_model is True,
                    then phi is an array of shape (num_dofs, ) 
                    with the configuration-space torques. 
                If consider_underactuation_model is False,
                    then phi is an array of shape (num_hysteresis, ) 
                    with the motor positions / twist angles of the proximal end of the rods.
            consider_underactuation_model (bool, optional): 
                If True, the underactuation model is considered. 
                Otherwise, the fully-actuated model is considered with the identity matrix as the actuation matrix.
            t0 (float, optionnal): Initial time.
                Default is 0.0.
            t1 (float, optionnal): Final time.
                Default is 10.0.
            dt (float, optionnal): Time step for the solver.
                Default is 1e-4.
            skip_steps (int, optionnal): Number of steps to skip in the output.
                This allows to reduce the number of saved time points.
                Default is 0.
            solver (AbstractSolver, optional): Solver to use for the ODE integration.
                Default is Tsit5() (Runge-Kutta 5(4) method).
            stepsize_controller (PIDController, optional): Stepsize controller for the solver.
                Default is ConstantStepSize().
            max_steps (int, optional): Maximum number of steps for the solver.
                Default is None (no limit).

        Returns:
            ts (Array): Time points at which the solution is saved.
            qs (Array): Configuration (strains) at the saved time points.
            qds (Array): Velocity (strains) at the saved time points.
        """
        y0 = jnp.concatenate([q0, qd0, jnp.zeros((self.num_hysteresis,))])

        term = ODETerm(self.forward_dynamics)

        t = jnp.arange(t0, t1, dt)  # Time points for the solution
        saveat = SaveAt(ts=t[::skip_steps])  # Save at specified time points

        # Prepare the actuation arguments
        actuation_args = (u0, control_fn, consider_underactuation_model)
        
        sol = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t[0],
            t1=t[-1],
            dt0=dt,
            y0=y0,
            args=actuation_args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )

        ts = sol.ts
        # Extract the configuration and velocity from the solution
        y_out = sol.ys
        qs, qds, zs = jnp.split(
            y_out, 
            [self.num_dofs, 2*self.num_dofs],
            axis=1
        )
        
        return ts, qs, qds