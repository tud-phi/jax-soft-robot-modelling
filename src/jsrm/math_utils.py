from jax import numpy as jnp
from jax import Array, lax, jit

@jit
def blk_diag(
    a: Array
) -> Array:
    """
    Create a block diagonal matrix from a tensor of blocks.

    Args:
        a: matrices to be block diagonalized of shape (m, n, o)

    Returns:
        b: block diagonal matrix of shape (m * n, m * o)

    """

    def assign_block_diagonal(i, _b):
        """
        Save the ith block  into the block-diagonal matrix `_b`
        Args:
            i: Index of block which we save into the block-diagonal matrix.
            _b: Block diagonal matrix. Should still have zeros at the ith block.
        Returns
        """
        # Assign the block saved in ith entry of `a` to the ith block-diagonal of `_b`
        # Hint: use `jax.lax.dynamic_update_slice` to update the entries of `_b`
        _b = lax.dynamic_update_slice(
            operand=_b, update=a[i], start_indices=(i * a.shape[1], i * a.shape[2])
        )
        return _b

    # Implement for loop to assign each block in `a` to the block-diagonal of `b`
    # Hint: use `jax.lax.fori_loop` and pass `assign_block_diagonal` as an argument
    b = jnp.zeros((a.shape[0] * a.shape[1], a.shape[0] * a.shape[2]), dtype=a.dtype)
    b = lax.fori_loop(
        lower=0,
        upper=a.shape[0],
        body_fun=assign_block_diagonal,
        init_val=b,
    )

    return b

@jit
def blk_concat(
    a: Array
) -> Array:
    """
    Concatenate horizontally (along the columns) a list of N matrices of size (a, b) to create a single matrix of size (a, b * N).

    Args:
        a (Array): matrices to be concatenated of shape (N, a, b)

    Returns:
        Array: concatenated matrix of shape (a, N * b)
    """
    b = a.transpose(1, 0, 2).reshape(a.shape[1], -1)
    return b

if __name__ == "__main__":
    # Example usage
    a = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print("Original array:")
    print(a)
    
    b = blk_diag(a)
    print("Block diagonal matrix:")
    print(b)
    
    c = blk_concat(a)
    print("Concatenated matrix:")
    print(c)