import jax
import jax.numpy as jnp
from jax import jit

# for documentation
from jax import Array
from typing import Sequence

@jit
def tilde_SE3( 
    vec3:Array
)-> Array:
    """
    Computes the tilde operator of SE(3) for a 3D vector.
    
    Args:
        vec (Array): array-like, shape (3,1)
            A 3-dimensional vector.
    
    Returns:
        Array: shape (3, 3)
            A 3x3 matrix representing the tilde operator of the input vector.
    """
    vec3 = vec3.reshape(-1)  # Ensure vec3 is a 1D array
    
    # Extract components of the vector
    x, y, z = vec3.flatten()
    
    # Use JAX's array creation for better performance
    Mtilde = jnp.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])
    return Mtilde

# def adjoint_SE2(
#     vec3:Array
# )-> Array:
#     """
#     Computes the adjoint representation of a vector of se(2).

#     Args:
#         vec (Array): array-like, shape (3,1)

#     Returns:
#         Array: shape (3, 3)
#             A 3x3 matrix representing the adjoint transformation of the input screw vector.
#     """
#     return tilde_SE3(vec3)  # The adjoint representation for se(2) is the same as the tilde operator

@jit
def adjoint_SE3(
    vec6:Array
)-> Array:
    """
    Computes the adjoint representation of a vector of se(3).

    Args:
        vec (Array): array-like, shape (6,1)
            A 6-dimensional vector representing the screw. 
            The first three elements correspond to the angular component, 
            and the last three elements correspond to the linear component.

    Returns:
        Array: shape (6, 6)
            A 6x6 matrix representing the adjoint transformation of the input screw vector.
    """
    vec6 = vec6.reshape(-1) # Ensure vec6 is a 1D array
    
    ang = vec6[:3].reshape((3, 1))  # Angular as a (3,1) vector
    lin = vec6[3:].reshape((3, 1))  # Linear as a (3,1) vector
    
    angtilde = tilde_SE3(ang)  # Tilde operator for angular part
    lintilde = tilde_SE3(lin)  # Tilde operator for linear part
    
    adj = jnp.block([
        [angtilde,        jnp.zeros((3, 3))],
        [lintilde,        angtilde]
    ])
    
    return adj

@jit
def hat_SE3(
    vec6:Array
)-> Array:
    """
    Computes the hat operator for a 6D vector of se(3).

    Args:
        vec (Array): array-like, shape (6,1)
            A 6-dimensional vector representing the screw. 
            The first three elements correspond to the angular component, 
            and the last three elements correspond to the linear component.

    Returns:
        Array: shape (4, 4)
            A 4x4 matrix representing the hat operator of the input screw vector.
    """
    vec6 = vec6.reshape(-1)  # Ensure vec6 is a 1D array
    
    ang = vec6[:3].reshape((3, 1))  # Angular as a (3,1) vector
    lin = vec6[3:].reshape((3, 1))  # Linear as a (3,1) vector
    
    angtilde = tilde_SE3(ang)  # Tilde operator for angular part
    
    hat = jnp.block([
        [angtilde, lin],
        [jnp.zeros((1, 3)), jnp.zeros((1, 1))] 
    ])
    
    return hat

@jit
def Adjoint_g_SE3(
    mat4:Array
)-> Array:
    """
    Computes the adjoint representation of a 4x4 matrix.
    
    Args:
        mat4 (Array): array-like, shape (4,4)
            A 4x4 matrix representing the transformation.
    
    Returns:
        Array: shape (4, 4)
            A 4x4 matrix representing the Adjoint transformation of the input matrix.
    """
    ang = mat4[:3, :3]  # Extract the angular part (top-left 3x3 block)
    lin = mat4[:3, 3].reshape((3, 1))  # Extract the linear part (top-right column)
    
    ltilde = tilde_SE3(lin)  # Tilde operator for linear part
    
    Adjoint = jnp.block([
        [ang, jnp.zeros((3, 3))],
        [ltilde @ ang, ang]
    ])
    
    return Adjoint

@jit
def Adjoint_gn_SE3(
    xi_n: Array,
    l_nprev: float,
    s :float,
    )-> Array:
    """
    Computes the adjoint representation of a position of a points at s (general curvilinear coordinate)
    along a rod in SE(3) deformed ine the current segment according to a strain vector xi_n.
    
    If s is a point of the n-th segment, this function use the lenght from the origin of the rod to the begining of the n-th segment.

    Args:
        xi_n (Array): array-like, shape (6,1)
            A 6-dimensional vector representing the screw in SE(3).
        l_nprev (float): 
            The length from the origin of the rod to the beginning of the n-th segment.
        s (float):
            The curvilinear coordinate along the rod, representing the position of a point in the n-th segment.

    Returns:
        Array: shape (6, 6)
            A 6x6 matrix representing the adjoint transformation of the input screw vector at the specified position.
    """
    # We suppose here that theta is not zero thanks to a previous use of apply_eps
    ang = xi_n[:3].reshape((3, 1))  # Angular as a (3,1) vector
    theta = jnp.linalg.norm(ang)  # Compute the norm of the angular part
    x = s-l_nprev  # Compute the segment length
    adjoint_xi_n = adjoint_SE3(xi_n)  # Adjoint representation of the input vector
    
    Adjoint = (jnp.eye(6)
            + 1/(2*theta) * (
                3*jnp.sin(x*theta) - x*theta*jnp.cos(x*theta)
                ) * adjoint_xi_n
            + 1/(2*jnp.power(theta, 2)) * (
                4 - 4*jnp.cos(x*theta) - x*theta*jnp.sin(x*theta)
                ) * jnp.linalg.matrix_power(adjoint_xi_n, 2)
            + 1/(2*jnp.power(theta, 3)) * (
                jnp.sin(x*theta) - x*theta*jnp.cos(x*theta)
                ) * jnp.linalg.matrix_power(adjoint_xi_n, 3)
            + 1/(2*jnp.power(theta, 4)) * (
                2 - 2*jnp.cos(x*theta) - x*theta*jnp.sin(x*theta)
                ) * jnp.linalg.matrix_power(adjoint_xi_n, 4))

    return Adjoint
    
@jit
def Tangent_gn_SE3(
    xi_n: Array,
    l_nprev: float,
    s :float,
    )-> Array:
    """
    Computes the tangent representation of a position of a points at s (general curvilinear coordinate)
    along a rod in SE(3) deformed in the current segment according to a strain vector xi_n.
    
    If s is a point of the n-th segment, this function use the length from the origin of the rod to the beginning of the n-th segment.

    Args:
        xi_n (Array): array-like, shape (6,1)
            A 6-dimensional vector representing the screw in SE(3).
        l_nprev (float): 
            The length from the origin of the rod to the beginning of the n-th segment.
        s (float): 
            The curvilinear coordinate along the rod, representing the position of a point in the n-th segment.

    Returns:
        Array: shape (6, 6)
            A 6x6 matrix representing the tangent transformation of the input screw vector at the specified position.
    """
    # We suppose here that theta is not zero thanks to a previous use of apply_eps
    ang = xi_n[:3].reshape((3, 1))  # Angular as a (3,1) vector
    theta = jnp.linalg.norm(ang)  # Compute the norm of the angular part
    x = s-l_nprev  # Compute the segment length
    adjoint_xi_n = adjoint_SE3(xi_n)  # Adjoint representation of the input vector
    
    Tangent = (x*jnp.eye(6)
            + 1/(2*jnp.power(theta, 2)) * (
                4 - 4*jnp.cos(x*theta) - x*theta*jnp.sin(x*theta)
                ) * adjoint_xi_n
            + 1/(2*jnp.power(theta, 3)) * (
                4*x*theta - 5*jnp.sin(x*theta) + x*theta*jnp.cos(x*theta)
                ) * jnp.linalg.matrix_power(adjoint_xi_n, 2)
            + 1/(2*jnp.power(theta, 4)) * (
                2 - 2*jnp.cos(x*theta) - x*theta*jnp.sin(x*theta)
                ) * jnp.linalg.matrix_power(adjoint_xi_n, 3)
            + 1/(2*jnp.power(theta, 5)) * (
                2*x*theta - 3*jnp.sin(x*theta) + x*theta*jnp.cos(x*theta)
                ) * jnp.linalg.matrix_power(adjoint_xi_n, 4))

    return Tangent

@jit
def vec_SE2_to_xi_SE3(
    vec3: Array, 
    indices: Sequence[int] = (2, 3, 4)
) -> Array:
    """
    Convert a strain vector in se(2) to a strain vector in se(3).

    Args:
        vec3 (Array): array-like, shape (3,1)
            A 3-dimensional vector representing the strain in se(2).
            The first element correspond to the angular component, 
            and the last elements corresponds to the linear component.
        indices (Sequence[int], optional): Indices in the 6D se(3) vector 
            where to insert the se(2) components. Default is (2, 3, 4)

    Returns:
        Array: shape (6,1)
            A 6-dimensional vector representing the strain in se(3).
            The first three elements correspond to the angular component, 
            and the last three elements correspond to the linear component.
    """
    vec3 = jnp.asarray(vec3).flatten()  # Ensure vec3 is a JAX array
    
    xi = jnp.zeros((6, ))  # Initialize a 6D vector with zeros
    xi = xi.at[jnp.array(indices)].set(vec3)  # Set the values at the specified indices
    return xi.reshape((6, 1))

@jit
def compute_weighted_sums(
    M: Array,
    vec6: Array,
    idx: int
)-> Array:
    """
    Compute the weighted sums of the matrix product of M and vec6,

    Args:
        M (Array): array of shape (N, 6, 6)
           Describes the matrix to be multiplied with vec6
        vec6 (Array): array-like of shape (N, 6)
           Describes the vector to be multiplied with M
        idx (int): index of the last row to be summed over

    Returns:
        Array: array of shape (N, 6)
           The result of the weighted sums. For each i, the result is the sum of the products of M[i, j] and vec6[j] for j from 0 to idx.
    """
    N = M.shape[0]
    # Matrix product for each j: (N, 6, 6) @ (N, 6, 1) -> (N, 6)
    prod = jnp.einsum('nij,nj->ni', M, vec6)

    # Triangular mask for partial sum: (N, N)
    # mask[i, j] = 1 if j >= i and j <= idx
    mask = (jnp.arange(N)[:, None] <= jnp.arange(N)[None, :]) & (jnp.arange(N)[None, :] <= idx)
    mask = mask.astype(M.dtype)  # (N, N)

    # Extend 6-dimensional mask (N, N, 1) to apply to (N, 6)
    masked_prod = mask[:, :, None] * prod[None, :, :]  # (N, N, 6)

    # Sum over j for each i : (N, 6)
    result = masked_prod.sum(axis=1) # (N, 6)

    return result