import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib import rc
rc('animation', html='html5')

from jsrm.systems import planar_pcs, planar_pcs_num
from pathlib import Path
import jsrm

# === Paramètres initiaux ===
num_segments = 1
rho = 1070 * jnp.ones((num_segments,))
params = {
    "th0": jnp.array(0.0),
    "l": 1e-1 * jnp.ones((num_segments,)),
    "r": 2e-2 * jnp.ones((num_segments,)),
    "rho": rho,
    "g": jnp.array([0.0, 9.81]),
    "E": 2e3 * jnp.ones((num_segments,)),
    "G": 1e3 * jnp.ones((num_segments,)),
}
params["D"] = 1e-3 * jnp.diag(
    (jnp.repeat(jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0)
     * params["l"][:, None]).flatten()
)
strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

# === Chargement des fonctions C ===
def get_C_fn(method, eps):
    if method == "symbolic":
        sym_exp_filepath = Path(jsrm.__file__).parent / "symbolic_expressions" / f"planar_pcs_ns-{num_segments}.dill"
        _, _, dyn_fn, _ = planar_pcs.factory(sym_exp_filepath, strain_selector)
    else:
        _, _, dyn_fn, _ = planar_pcs_num.factory(
            num_segments,
            strain_selector,
            integration_type="gauss-legendre",
            param_integration=5,
            jacobian_type=method
        )
    def C_fn(params, q, q_d, eps=None):
        B, C, G, K, D, alpha = dyn_fn(params, q, q_d, eps)
        return C
    return C_fn

# === Options d'affichage ===
C_colors = {"symbolic": "green", "explicit": "orange", "autodiff": "blue"}
C_markers = {"symbolic": "s", "explicit": "x", "autodiff": "o"}
C_types = ["symbolic", "explicit", "autodiff"]
active_C_types = C_types.copy()

# === eps ===
eps_options = [None, 1e-6, 1e-5, 1e-4]
eps_labels = ['None'] + [f'1e-{i}' for i in range(6, 2, -1)]
def get_eps_from_slider():
    return eps_options[int(eps_slider.val)]

# === Espaces des paramètres ===
kappa_values = jnp.linspace(-0.1, 0.1, 50)
sigma_x, sigma_y = 0.05, 1.05  # Note: on soustraira 1 à sigma_y plus tard

# === Subplots ===
fig, axs = plt.subplots(3, 3, figsize=(15, 8))

def plot_C(kappa_vals, sigma_x, sigma_y, eps, fig, axs):
    for ax in axs.flat:
        ax.clear()
        ax.set_xlabel("kappa")
        ax.set_ylabel("C[i,j]")
        ax.grid(True)

    for method in active_C_types:
        C_fn = {
            "symbolic": get_C_fn("symbolic", eps),
            "explicit": get_C_fn("explicit", eps),
            "autodiff": get_C_fn("autodiff", eps),
        }[method]

        Cs = []
        for kappa in kappa_vals:
            q = jnp.array([kappa, sigma_x, sigma_y - 1.0] * num_segments)
            q_d = jnp.ones_like(q)  # valeur arbitraire
            Cs.append(C_fn(params, q, q_d, eps=eps))
        Cs = jnp.stack(Cs)

        for i in range(3):
            for j in range(3):
                axs[i, j].plot(
                    kappa_vals,
                    Cs[:, i, j],
                    label=method,
                    color=C_colors[method],
                    marker=C_markers[method],
                    alpha=0.8,
                    markerfacecolor='none',
                    markeredgecolor=C_colors[method]
                )
                axs[i, j].set_title(f'C[{i},{j}]')

    fig.suptitle(f"Matrice de Coriolis C selon kappa (σx={sigma_x:.2f}, σy={sigma_y-1:.2f}, eps={eps})")
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.87, 0.5))

# === Premier tracé ===
plot_C(kappa_values, sigma_x, sigma_y, eps_options[0], fig, axs)

# === Sliders ===
sigma_x_slider = Slider(plt.axes([0.2, 0.01, 0.65, 0.03]), 'σx', 0.0, 0.1, valinit=sigma_x)
sigma_y_slider = Slider(plt.axes([0.2, 0.05, 0.65, 0.03]), 'σy', 1.0, 1.1, valinit=sigma_y)
eps_slider = Slider(plt.axes([0.2, 0.09, 0.65, 0.03]), 'eps (log)', 0, len(eps_options) - 1, valinit=0, valstep=1)

def update_plot(_):
    sx = sigma_x_slider.val
    sy = sigma_y_slider.val
    eps = get_eps_from_slider()
    plot_C(kappa_values, sx, sy, eps, fig, axs)
    fig.canvas.draw_idle()

sigma_x_slider.on_changed(update_plot)
sigma_y_slider.on_changed(update_plot)
eps_slider.on_changed(update_plot)

# === Boutons ===
reset_ax = plt.axes([0.87, 0.6, 0.1, 0.05])
reset_button = Button(reset_ax, 'Reset sliders')
reset_button.on_clicked(lambda event: (
    sigma_x_slider.reset(), sigma_y_slider.reset(), eps_slider.reset()))

# === Checkboxes ===
check_ax = plt.axes([0.87, 0.7, 0.12, 0.15])
check = CheckButtons(check_ax, C_types, [True] * len(C_types))
check_ax.set_title("Méthodes C")

def on_check(label):
    global active_C_types
    active_C_types = [C_types[i] for i, v in enumerate(check.get_status()) if v]
    update_plot(None)

check.on_clicked(on_check)

plt.show()
