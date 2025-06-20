import jax
import jax.numpy as jnp

# for documentation
from jax import Array
from typing import Sequence

def tilde_SE3( 
    vec3:Array
)-> Array:
    """
    Computes the tilde operator of SE(3) for a 3D vector.
    
    Args:
        vec3 (Array): array-like, shape (3,1)
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

def adjoint_SE3(
    vec6:Array
)-> Array:
    """
    Computes the adjoint representation of a vector of se(3).

    Args:
        vec6 (Array): array-like, shape (6,1)
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

def adjoint_star_SE3(
    vec6:Array
)-> Array:
    """
    Computes the co-adjoint representation of a vector of se(3).

    Args:
        vec6 (Array): array-like, shape (6,1)
            A 6-dimensional vector representing the screw. 
            The first three elements correspond to the angular component, 
            and the last three elements correspond to the linear component.

    Returns:
        Array: shape (6, 6)
            A 6x6 matrix representing the co-adjoint transformation of the input screw vector.
    """
    vec6 = vec6.reshape(-1) # Ensure vec6 is a 1D array
    
    ang = vec6[:3].reshape((3, 1))  # Angular as a (3,1) vector
    lin = vec6[3:].reshape((3, 1))  # Linear as a (3,1) vector
    
    angtilde = tilde_SE3(ang)  # Tilde operator for angular part
    lintilde = tilde_SE3(lin)  # Tilde operator for linear part
    
    adj_star = jnp.block([
        [angtilde           , lintilde],
        [jnp.zeros((3, 3))  , angtilde]
    ])
    
    return adj_star

def hat_SE3(
    vec6:Array
)-> Array:
    """
    Computes the hat operator for a 6D vector of se(3).

    Args:
        vec6 (Array): array-like, shape (6,1)
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

def Adjoint_gn_SE3_inv(
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
    
    # Extract R and uR from the Adjoint matrix
    R = Adjoint[:3, :3]
    uR = Adjoint[3:, :3]

    # Compute the inverse using the Schur complement
    R_inv = jnp.transpose(R)    # Since R is a rotation matrix
    u = jnp.dot(uR, R_inv)      # Compute the linear part
    uR_inv = -jnp.dot(R_inv, u)  # Compute the inverse linear part

    # Construct the inverse Adjoint matrix
    inverse_Adjoint = jnp.block([
        [R_inv, jnp.zeros((3, 3))],
        [uR_inv, R_inv]
    ])

    return inverse_Adjoint

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

# ================================================================================================
# SE(2) operators
# ===================================
J = jnp.array([[0, -1], [1, 0]])

def adjoint_SE2(
    vec3:Array
)-> Array:
    """
    Computes the adjoint representation of a vector of se(2).

    Args:
        vec3 (Array): array-like, shape (3, 1)
            A 3-dimensional vector representing the screw. 
            The first element correspond to the angular component, 
            and the last two elements correspond to the linear component.

    Returns:
        Array: shape (3, 3)
            A 3x3 matrix representing the adjoint transformation of the input screw vector.
    """
    vec3 = vec3.reshape(-1) # Ensure vec6 is a 1D array
    
    ang = vec3[0]
    lin = vec3[1:].reshape((2, 1))  # Linear as a (3,1) vector
    
    adj = jnp.concatenate([
        jnp.zeros((1, 3)),
        jnp.concatenate([-J@lin, ang*J], axis=1)
    ])
    
    return adj

def adjoint_star_SE2(
    vec3:Array
)-> Array:
    """
    Computes the co-adjoint representation of a vector of se(2).

    Args:
        vec3 (Array): array-like, shape (3, 1)
            A 3-dimensional vector representing the screw. 
            The first element correspond to the angular component, 
            and the last two elements correspond to the linear component.

    Returns:
        Array: shape (3, 3)
            A 3x3 matrix representing the co-adjoint transformation of the input screw vector.
    """
    vec3 = vec3.reshape(-1) # Ensure vec6 is a 1D array
    
    ang = vec3[0]
    lin = vec3[1:].reshape((2, 1))  # Linear as a (3,1) vector
    
    adj_star = jnp.concatenate([
        jnp.zeros((3, 1)),
        jnp.concatenate([lin.T@J, ang*J], axis=0)
    ], axis=1)
    
    return adj_star

def Adjoint_g_SE2(
    mat3:Array
)-> Array:
    """
    Computes the adjoint representation of a 3x3 matrix.
    
    Args:
        mat4 (Array): array-like, shape (4,4)
            A 4x4 matrix representing the transformation.
    
    Returns:
        Array: shape (3, 3)
            A 3x3 matrix representing the Adjoint transformation of the input matrix.
    """
    R = mat3[:2, :2]  # Extract the angular part (top-left 2x2 block)
    t = mat3[:2, 2].reshape((2, 1))  # Extract the linear part (top-right column)
    
    Adjoint = jnp.concatenate([
        jnp.concatenate([jnp.ones(((1,1))), jnp.zeros((1, 2))], axis=1), 
        jnp.concatenate([-J @ t, R], axis=1)
    ])
    
    return Adjoint

def Adjoint_gn_SE2(
    xi_n: Array,
    l_nprev: float,
    s :float,
    )-> Array:
    """
    Computes the adjoint representation of a position of a points at s (general curvilinear coordinate)
    along a rod in SE(2) deformed ine the current segment according to a strain vector xi_n.
    
    If s is a point of the n-th segment, this function use the lenght from the origin of the rod to the begining of the n-th segment.

    Args:
        xi_n (Array): array-like, shape (3,1)
            A 3-dimensional vector representing the screw in SE(2).
        l_nprev (float): 
            The length from the origin of the rod to the beginning of the n-th segment.
        s (float):
            The curvilinear coordinate along the rod, representing the position of a point in the n-th segment.

    Returns:
        Array: shape (3, 3)
            A 3x3 matrix representing the adjoint transformation of the input screw vector at the specified position.
    """
    # We suppose here that theta is not zero thanks to a previous use of apply_eps
    theta = xi_n[0]  # Angular part
    x = s-l_nprev  # Compute the segment length
    adjoint_xi_n = adjoint_SE2(xi_n)  # Adjoint representation of the input vector
    
    Adjoint = (jnp.eye(3)
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

def Adjoint_gn_SE2_inv(
    xi_n: Array,
    l_nprev: float,
    s :float,
    )-> Array:
    """
    Computes the adjoint representation of a position of a points at s (general curvilinear coordinate)
    along a rod in SE(2) deformed ine the current segment according to a strain vector xi_n.
    
    If s is a point of the n-th segment, this function use the lenght from the origin of the rod to the begining of the n-th segment.

    Args:
        xi_n (Array): array-like, shape (3,1)
            A 3-dimensional vector representing the screw in SE(2).
        l_nprev (float): 
            The length from the origin of the rod to the beginning of the n-th segment.
        s (float):
            The curvilinear coordinate along the rod, representing the position of a point in the n-th segment.

    Returns:
        Array: shape (3, 3)
            A 3x3 matrix representing the adjoint transformation of the input screw vector at the specified position.
    """
    # We suppose here that theta is not zero thanks to a previous use of apply_eps
    theta = xi_n[0]  # Angular part
    x = s-l_nprev  # Compute the segment length
    adjoint_xi_n = adjoint_SE2(xi_n)  # Adjoint representation of the input vector
    
    Adjoint = (jnp.eye(3)
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
    
    # Extract R and -Jt from the Adjoint matrix
    R = Adjoint[1:, 1:]
    mJt = Adjoint[1:, 0].reshape(-1, 1)

    # Compute the inverse using the Schur complement
    R_inv = jnp.transpose(R)    # Since R is a rotation matrix, R^-1=R^T
    # Construct the inverse Adjoint matrix
    inverse_Adjoint = jnp.concatenate([
        jnp.concatenate([jnp.ones(((1,1))), jnp.zeros((1, 2))], axis=1), 
        jnp.concatenate([-R_inv@mJt, R_inv], axis=1)
    ])

    return inverse_Adjoint

def Tangent_gn_SE2(
    xi_n: Array,
    l_nprev: float,
    s :float,
    )-> Array:
    """
    Computes the tangent representation of a position of a points at s (general curvilinear coordinate)
    along a rod in SE(2) deformed in the current segment according to a strain vector xi_n.
    
    If s is a point of the n-th segment, this function use the length from the origin of the rod to the beginning of the n-th segment.

    Args:
        xi_n (Array): array-like, shape (3,1)
            A 3-dimensional vector representing the screw in SE(2).
        l_nprev (float): 
            The length from the origin of the rod to the beginning of the n-th segment.
        s (float): 
            The curvilinear coordinate along the rod, representing the position of a point in the n-th segment.

    Returns:
        Array: shape (3, 3)
            A 3x3 matrix representing the tangent transformation of the input screw vector at the specified position.
    """
    # We suppose here that theta is not zero thanks to a previous use of apply_eps
    theta = xi_n[0]  # Angular part
    x = s-l_nprev  # Compute the segment length
    adjoint_xi_n = adjoint_SE2(xi_n)  # Adjoint representation of the input vector
    
    Tangent = (x*jnp.eye(3)
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

# ================================================================================================
# Shared operators
# ============================
def compute_weighted_sums(
    M: Array,
    vecm: Array,
    idx: int
)-> Array:
    """
    Compute the weighted sums of the matrix product of M and vecm,

    Args:
        M (Array): array of shape (N, m, m)
           Describes the matrix to be multiplied with vecm
        vecm (Array): array-like of shape (N, m)
           Describes the vector to be multiplied with M
        idx (int): index of the last row to be summed over

    Returns:
        Array: array of shape (N, m)
           The result of the weighted sums. For each i, the result is the sum of the products of M[i, j] and vecm[j] for j from 0 to idx.
    """
    N = M.shape[0]
    # Matrix product for each j: (N, m, m) @ (N, m, 1) -> (N, m)
    prod = jnp.einsum('nij,nj->ni', M, vecm)

    # Triangular mask for partial sum: (N, N)
    # mask[i, j] = 1 if j >= i and j <= idx
    mask = (jnp.arange(N)[:, None] <= jnp.arange(N)[None, :]) & (jnp.arange(N)[None, :] <= idx)
    mask = mask.astype(M.dtype)  # (N, N)

    # Extend 6-dimensional mask (N, N, 1) to apply to (N, m)
    masked_prod = mask[:, :, None] * prod[None, :, :]  # (N, N, m)

    # Sum over j for each i : (N, m)
    result = masked_prod.sum(axis=1) # (N, m)
    return result