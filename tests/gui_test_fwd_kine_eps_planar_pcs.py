import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib import rc

rc("animation", html="html5")

from jsrm.systems import planar_pcs_num, planar_pcs_sym
from pathlib import Path
import jsrm
from tqdm import tqdm
from matplotlib.animation import FFMpegWriter

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
    (
        jnp.repeat(jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0)
        * params["l"][:, None]
    ).flatten()
)
strain_selector = jnp.ones((3 * num_segments,), dtype=bool)


# === Chargement des fonctions Jacobiennes ===
def get_fwd_kine_fn(jacobian_type):
    if jacobian_type == "symbolic":
        sym_exp_filepath = (
            Path(jsrm.__file__).parent
            / "symbolic_expressions"
            / f"planar_pcs_ns-{num_segments}.dill"
        )
        _, fwd, _, _ = planar_pcs_sym.factory(sym_exp_filepath, strain_selector)
    else:
        _, fwd, _, _ = planar_pcs_num.factory(
            num_segments,
            strain_selector,
            integration_type="gauss-legendre",
            param_integration=5,
            jacobian_type=jacobian_type,
        )
    return fwd


FwdKine_autodiff_fn = get_fwd_kine_fn("autodiff")
FwdKine_explicit_fn = get_fwd_kine_fn("explicit")
FwdKine_symbolic_fn = get_fwd_kine_fn("symbolic")

jacobian_colors = {"symbolic": "green", "explicit": "orange", "autodiff": "blue"}
jacobian_markers = {"symbolic": "s", "explicit": "x", "autodiff": "o"}
jacobian_types = ["symbolic", "explicit", "autodiff"]
list_of_type_of_jacobian = jacobian_types.copy()

# === eps discret ===
eps_options = [None, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
eps_labels = ["None"] + [f"1e-{i}" for i in range(6, 0, -1)]


def get_eps_from_slider():
    return eps_options[int(eps_slider.val)]


# === Variables physiques ===
borne_kappa = 1e-3
nb_kappa = 51
kappa_values = jnp.linspace(-borne_kappa, borne_kappa, nb_kappa)

borne_sigma_x, borne_sigma_y = 1e-1, 1e-1
nb_sigma_x, nb_sigma_y, nb_s = 50, 50, 50
sigma_x_values = jnp.linspace(0, borne_sigma_x, nb_sigma_x)
sigma_y_values = jnp.linspace(0, borne_sigma_y, nb_sigma_y)
s_values = jnp.linspace(0, float(params["l"][0]), nb_s)

# === Tracé principal ===
fig, axs = plt.subplots(2, figsize=(15, 8))


def FwdKine_plot(eps_list, s, sigma_x, sigma_y, fig, axs):
    for ax in axs:
        ax.clear()
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        ax.set_xlabel("kappa (bending strain)")
        ax.set_ylabel("Fwd Kinematics components")

    for i_eps, eps in enumerate(eps_list):
        FwdKine_kwargs = {"eps": eps} if eps is not None else {}
        FwdKine_auto, FwdKine_exp, FwdKine_symb = [], [], []
        for kappa in kappa_values:
            q = jnp.array([kappa, sigma_x, sigma_y - 1.0] * num_segments)
            FwdKine_auto.append(FwdKine_autodiff_fn(params, q, s, **FwdKine_kwargs))
            FwdKine_exp.append(FwdKine_explicit_fn(params, q, s, **FwdKine_kwargs))
            FwdKine_symb.append(FwdKine_symbolic_fn(params, q, s, **FwdKine_kwargs))
        FwdKine_auto, FwdKine_exp, FwdKine_symb = (
            jnp.stack(FwdKine_auto),
            jnp.stack(FwdKine_exp),
            jnp.stack(FwdKine_symb),
        )

        for i in range(2):
            if eps is not None:
                axs[i].axvline(
                    eps,
                    color="red",
                    linestyle=":",
                    linewidth=2,
                    alpha=(i_eps + 1) / len(eps_list),
                    label=f"+/-eps={eps:.2e}",
                )
                axs[i].axvline(
                    -eps,
                    color="red",
                    linestyle=":",
                    linewidth=2,
                    alpha=(i_eps + 1) / len(eps_list),
                )

            for j_type in list_of_type_of_jacobian:
                FwdKine = {
                    "symbolic": FwdKine_symb,
                    "explicit": FwdKine_exp,
                    "autodiff": FwdKine_auto,
                }[j_type]
                axs[i].plot(
                    kappa_values,
                    FwdKine[:, i],
                    marker=jacobian_markers[j_type],
                    label=f"FwdKine_{j_type}",
                    color=jacobian_colors[j_type],
                    alpha=(i_eps + 1) / len(eps_list),
                    markerfacecolor="none",
                    markeredgecolor=jacobian_colors[j_type],
                )
            axs[i].set_title(f"FwdKine[{i}]")
            axs[i].grid(True)

    fig.suptitle(
        f"Fwd Kinematics components as a function of kappa\ns = {s:.3f}, sigma_x = {sigma_x:.3f}, sigma_y = {sigma_y - 1:.3f}"
    )
    param_legend = 0.85
    fig.tight_layout(rect=[0, 0, param_legend, 1])

    handles, labels = axs[0].get_legend_handles_labels()
    # Supprimer doublons
    unique = dict(zip(labels, handles))
    handles, labels = list(unique.values()), list(unique.keys())
    handles += [Patch(facecolor="white")]
    labels += [f"s = {s:.2f}"]
    if fig.legends:
        for leg in fig.legends:
            leg.remove()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(param_legend, 0.5))


# === Valeurs initiales ===
initial_s = float(params["l"][0] / 2)
initial_sigma_x = float(borne_sigma_x / 2)
initial_sigma_y = float(borne_sigma_y / 2)

# === Première visualisation ===
FwdKine_plot([eps_options[0]], initial_s, initial_sigma_x, initial_sigma_y, fig, axs)

# === Sliders ===
s_slider = Slider(
    plt.axes([0.2, 0.01, 0.65, 0.03]),
    "s",
    float(s_values[0]),
    float(s_values[-1]),
    valinit=initial_s,
)
sigma_x_slider = Slider(
    plt.axes([0.2, 0.05, 0.65, 0.03]),
    "sigma_x",
    float(sigma_x_values[0]),
    float(sigma_x_values[-1]),
    valinit=initial_sigma_x,
)
sigma_y_slider = Slider(
    plt.axes([0.2, 0.09, 0.65, 0.03]),
    "sigma_y",
    float(sigma_y_values[0]),
    float(sigma_y_values[-1]),
    valinit=initial_sigma_y,
)
eps_slider = Slider(
    plt.axes([0.2, 0.13, 0.65, 0.03]),
    "eps (log scale)",
    0,
    len(eps_options) - 1,
    valinit=0,
    valstep=1,
)


def on_slider_change(s_val, sigma_x_val, sigma_y_val):
    s = float(s_val)
    sigma_x = float(sigma_x_val)
    sigma_y = float(sigma_y_val)
    eps_val = get_eps_from_slider()
    FwdKine_plot([eps_val], s, sigma_x, sigma_y, fig, axs)
    fig.canvas.draw_idle()


def update_sliders(_):
    on_slider_change(s_slider.val, sigma_x_slider.val, sigma_y_slider.val)


s_slider.on_changed(update_sliders)
sigma_x_slider.on_changed(update_sliders)
sigma_y_slider.on_changed(update_sliders)
eps_slider.on_changed(update_sliders)

# === Boutons et CheckBoxes ===
reset_ax = plt.axes([0.87, 0.6, 0.1, 0.05])
reset_button = Button(reset_ax, "Reset sliders")
reset_button.on_clicked(
    lambda event: (
        s_slider.reset(),
        sigma_x_slider.reset(),
        sigma_y_slider.reset(),
        eps_slider.reset(),
    )
)

check_ax = plt.axes([0.87, 0.7, 0.12, 0.15])
check = CheckButtons(check_ax, jacobian_types, [True] * len(jacobian_types))
check_ax.set_title("Jacobian types")


def on_check(label):
    global list_of_type_of_jacobian
    list_of_type_of_jacobian = [
        jacobian_types[i] for i, v in enumerate(check.get_status()) if v
    ]
    update_sliders(None)


check.on_clicked(on_check)

plt.show()
