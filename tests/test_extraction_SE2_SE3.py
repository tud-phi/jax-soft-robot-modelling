import jax.numpy as jnp

# Nombre de segments
n_segments = 2

# Construction manuelle : chaque segment aura une matrice 6x6 différente
J_full = jnp.array([
    [[ 0,  1,  2,  3,  4,  5],
     [10, 11, 12, 13, 14, 15],
     [20, 21, 22, 23, 24, 25],
     [30, 31, 32, 33, 34, 35],
     [40, 41, 42, 43, 44, 45],
     [50, 51, 52, 53, 54, 55]],

    [[100,101,102,103,104,105],
     [110,111,112,113,114,115],
     [120,121,122,123,124,125],
     [130,131,132,133,134,135],
     [140,141,142,143,144,145],
     [150,151,152,153,154,155]]
])  # shape: (2, 6, 6)

# # Indices d’intérêt
# interest_coordinates = [2, 3, 4]
# interest_strain = [1, 2, 3]

# # Extraction réduite
# J_reduced = J_full[:, interest_coordinates][:, :, interest_strain]  # shape: (2, 3, 3)

# Affichage    
print("\nJacobienne complète (2 segments, 6x6):\n", J_full)
# print("Jacobienne réduite (2 segments, 3x3):\n", J_reduced)

# reordered_columns = [1, 2, 0]
# J_final = J_reduced[:, reordered_columns, :]

# # Affichage de la jacobienne finale
# print("\nJacobienne finale (2 segments, 3x3) après réarrangement :\n", J_final)

interest_coordinates = [2, 3, 4]
interest_strain = [2, 3, 4]
reordered_columns = [1, 2, 0]
J_tout_dun_coup = J_full[:, interest_coordinates][:, :, interest_strain][:, reordered_columns, :]

# Affichage de la jacobienne finale avec extraction et réarrangement en une seule étape
print("\nJacobienne finale (2 segments, 3x3) avec extraction et réarrangement :\n", J_tout_dun_coup)