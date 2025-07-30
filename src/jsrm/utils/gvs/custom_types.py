from typing import Literal, Union, List, Optional, Tuple
from jax import Array

import jax
from dataclasses import dataclass, field


@dataclass
class LinkAttributes:
    section: Literal['Circular', 'Rectangular', 'Elliptical']
    E: float
    nu: float
    rho: float
    eta: float
    l: float

    r_i: Optional[float] = 0.0
    r_f: Optional[float] = 0.0
    h_i: Optional[float] = 0.0
    h_f: Optional[float] = 0.0
    w_i: Optional[float] = 0.0
    w_f: Optional[float] = 0.0
    a_i: Optional[float] = 0.0
    a_f: Optional[float] = 0.0
    b_i: Optional[float] = 0.0
    b_f: Optional[float] = 0.0

@dataclass
class JointAttributes:
    jointtype: Literal['Revolute', 'Prismatic', 'Helical', 'Cylindrical', 'Planar', 'Spherical', 'Free', 'Fixed']

    axis    : Literal['x', 'y', 'z']    = 'x'
    plane   : Literal['xy', 'yz', 'xz'] = 'xy'
    pitch   : float = 0.0
    
    K_joint : Union[Array, List] = field(default_factory=list)

@dataclass
class BasisAttributes:
    basistype   : Literal['Monomial', 'Legendre', 'Chebychev', 'Fourier', 'Gaussian', 'IMQ']
    Bdof        : Union[Array, List]  # shape (6,1) indicating whether each type of deformation is selected (1) or not (0)
    Bodr        : Union[Array, List]  # shape (6,1) indicating the orders of the basis functions for each type of deformation
    xi_star     : Union[Array, List]  # shape (6,1) indicating the reference strain values for each type of deformation

@jax.tree_util.register_pytree_node_class
@dataclass
class JointOperand:
    axis_idx    : int
    pitch       : float
    plane_idx   : int
    
    def tree_flatten(self):
        children = (self.axis_idx, self.pitch, self.plane_idx)
        aux_data = None  # aucun champ statique à exclure ici
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@jax.tree_util.register_pytree_node_class
@dataclass
class GeometricOperand:
    Xs      : Array
    r_params: Tuple[Array, Array]
    h_params: Tuple[Array, Array]
    w_params: Tuple[Array, Array]
    a_params: Tuple[Array, Array]
    b_params: Tuple[Array, Array]
    
    def tree_flatten(self):
        children = (self.Xs, self.r_params, self.h_params, self.w_params, self.a_params, self.b_params)
        aux_data = None  # aucun champ statique à exclure ici
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@jax.tree_util.register_pytree_node_class
@dataclass
class SegmentData:
    l               : Array # Length of the segment
    nip             : Array # Number of integration points
    dofs_joint_link : Array # Degrees of freedom of the segment as [dof_joint, dof_link]
    strain_selector : Array # Boolean array indicating which strain components are active
    Xs              : Array # Integration points
    Ws              : Array # Weights for the integration points
    Ms              : Array # Mass matrices at integration points
    Es              : Array # Stiffness matrices at integration points
    Gs              : Array # Damping matrices at integration points
    B_joint         : Array # Joint basis matrix
    B_Xs            : Array # Basis matrix at integration points
    B_Z1            : Array # Basis matrix at Z1 points
    B_Z2            : Array # Basis matrix at Z2 points
    xi_star_joint   : Array # Joint initial strain vector
    xi_star_Xs      : Array # Initial strain vector at integration points
    xi_star_Z1      : Array # Initial strain vector at Z1 points
    xi_star_Z2      : Array # Initial strain vector at Z2 points
    K_joint         : Array # Joint stiffness matrix
    
    def tree_flatten(self):
        children = (
            self.l, self.nip, self.dofs_joint_link, self.Xs, self.Ws,
            self.Ms, self.Es, self.Gs, self.B_joint, self.B_Xs, self.B_Z1,
            self.B_Z2, self.xi_star_joint, self.xi_star_Xs, self.xi_star_Z1,
            self.xi_star_Z2, self.K_joint
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
@jax.tree_util.register_pytree_node_class
@dataclass
class LinkData:
    l               : Array # Length of the segment
    nip             : Array # Number of integration points
    strain_selector : Array # Boolean array indicating which strain components are active
    Xs              : Array # Integration points
    Ws              : Array # Weights for the integration points
    Ms              : Array # Mass matrices at integration points
    Es              : Array # Stiffness matrices at integration points
    Gs              : Array # Damping matrices at integration points
    B_Xs            : Array # Basis matrix at integration points
    B_Z1            : Array # Basis matrix at Z1 points
    B_Z2            : Array # Basis matrix at Z2 points
    xi_star_Xs      : Array # Initial strain vector at integration points
    xi_star_Z1      : Array # Initial strain vector at Z1 points
    xi_star_Z2      : Array # Initial strain vector at Z2 points
    dof_link        : Array # Degrees of freedom of the segment
    
    def tree_flatten(self):
        children = (
            self.l, 
            self.nip, 
            self.Xs, 
            self.Ws,
            self.Ms, 
            self.Es, 
            self.Gs, 
            self.B_Xs, 
            self.B_Z1,
            self.B_Z2, 
            self.xi_star_Xs, 
            self.xi_star_Z1,
            self.xi_star_Z2, 
            self.dof_link,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
  