import jax
import jax.numpy as jnp
from jax import grad, jit, lax, vmap
import timeit

# Problème : matrice B simulée pour benchmark
n = 10
params = {}
xi = jnp.arange(1, n + 1).astype(jnp.float32)
xi_d = jnp.arange(1, n + 1).astype(jnp.float32)

def B_fn_xi(params, xi):
    return jnp.outer(xi, xi) + jnp.eye(n)

# --------- Version 1 : triple loop via lax.fori_loop ----------
def compute_C_loop(params, xi, xi_d):
    B = B_fn_xi(params, xi)
    C = jnp.zeros_like(B)

    def body_i(i, C):
        def body_j(j, C):
            def body_k(k, acc):
                dB_ij = grad(lambda x: B_fn_xi(params, x)[i, j])(xi)[k]
                dB_ik = grad(lambda x: B_fn_xi(params, x)[i, k])(xi)[j]
                dB_jk = grad(lambda x: B_fn_xi(params, x)[j, k])(xi)[i]
                coeff = 0.5 * (dB_ij + dB_ik - dB_jk) * xi_d[k]
                return acc + coeff
            C_ij = lax.fori_loop(0, n, body_k, 0.0)
            return C.at[i, j].set(C_ij)
        return lax.fori_loop(0, n, body_j, C)
    return lax.fori_loop(0, n, body_i, C)

# --------- Version 2 : vmap ----------
def compute_C_vmap(params, xi, xi_d):
    B = B_fn_xi(params, xi)
    def christoffel_fn(i, j, k):
        return 0.5 * (
            grad(lambda x: B_fn_xi(params, x)[i, j])(xi)[k]
            + grad(lambda x: B_fn_xi(params, x)[i, k])(xi)[j]
            - grad(lambda x: B_fn_xi(params, x)[j, k])(xi)[i]
        )
    def C_ij(i, j):
        cs_k = vmap(lambda k: christoffel_fn(i, j, k))(jnp.arange(n))
        return jnp.dot(cs_k, xi_d)
    return vmap(lambda i: vmap(lambda j: C_ij(i, j))(jnp.arange(n)))(jnp.arange(n))

# --------- JIT functions ---------
jit_loop = jit(compute_C_loop)
jit_vmap = jit(compute_C_vmap)

# Warmup (JAX compile)
jit_loop(params, xi, xi_d)
jit_vmap(params, xi, xi_d)

# --------- timeit benchmark ---------
repeat = 3
number = 10

t_loop = timeit.timeit(lambda: jit_loop(params, xi, xi_d).block_until_ready(), number=number) / number
t_vmap = timeit.timeit(lambda: jit_vmap(params, xi, xi_d).block_until_ready(), number=number) / number

print(f"\nTemps d'évaluation (moyenne sur {number} exécutions):")
print(f"Loop version : {t_loop*1000:.2e} ms")
print(f"Vmap version : {t_vmap*1000:.2e} ms")

# --------- Comparaison des valeurs de C ---------
C_loop = jit_loop(params, xi, xi_d)
C_vmap = jit_vmap(params, xi, xi_d)

diff = jnp.max(jnp.abs(C_loop - C_vmap))
print(f"\nDifférence maximale entre les deux versions de C : {diff:.2e}")
