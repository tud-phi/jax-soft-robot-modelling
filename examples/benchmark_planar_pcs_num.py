import cv2  # importing cv2
from functools import partial
import jax

jax.config.update("jax_enable_x64", True)  # double precision
from diffrax import diffeqsolve, Euler, ODETerm, SaveAt, Tsit5
from jax import Array, vmap
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from typing import Callable, Dict

import jsrm
from jsrm import ode_factory
from jsrm.systems import planar_pcs, planar_pcs_num

import time
import pickle

import timeit
import statistics

def simulate_planar_pcs(
    num_segments: int,
    type_of_derivation: str = "symbolic",
    type_of_integration: str = "gauss-legendre",
    param_integration: int = None,
    type_of_jacobian: str = "explicit",
    robot_params: Dict[str, Array] = None,
    strain_selector: Array = None,
    q0: Array = None,
    q_d0: Array = None,
    t: float = 1.0,
    dt: float = None,
    bool_print: bool = True,
    bool_plot: bool = True,
    bool_save_plot: bool = False,
    bool_save_video: bool = True,
    bool_save_res: bool = False,
    results_path: str = None, 
    results_path_extension: str = None
) -> Dict:
    """
    Simulate a planar PCS model. Save the video and figures.

    Args:
        num_segments (int): number of segments of the robot.
        type_of_derivation (str, optional): type of derivation ("symbolic" or "numeric"). 
            Defaults to "symbolic".
        type_of_integration (str, optional): scheme of integration ("gauss-legendre", "gauss-kronrad" or "trapezoid"). 
            Defaults to "gauss-legendre".
        param_integration (int, optional): parameter for the integration scheme (number of points for gauss or number of points for trapezoid). 
            Defaults to 1000 for trapezoid and 5 for gauss.
        type_of_jacobian (str, optional): type of jacobian ("explicit" or "autodiff").
            Defaults to "explicit".
        strain_selector (Array, optional): selector for the strains as a boolean array.
            Defaults to all.
        robot_params (Dict[str, Array], optional): parameters of the robot. 
        strain_selector (Array, optional): selector for the strains.
        t (float, optional): time of simulation [s]. 
            Defaults to 1.0s.
        dt (float, optional): time step [s]. 
            Defaults to 1e-4s.
        q0 (Array, optional): initial configuration.
            Defaults to a configuration with 5.0pi rad of bending, 0.1 of shear, and 0.2 of axial strain for each segment.
        q_d0 (Array, optional): initial velocity.
            Defaults to 0 array.
        bool_print (bool, optional): if True, print the simulation results.
            Defaults to True.
        bool_plot (bool, optional): if True, show the figures. 
            Defaults to True.
        bool_save_plot (bool, optional): if True, save the figures.
            Defaults to False.
        bool_save_video (bool, optional): if True, save the video.
            Defaults to False.
        bool_save_res (bool, optional): if True, save the results of the simulation. 
            Defaults to False.
        results_path (str, optional): path to save the dictionary with the simulation results. Must have the suffix .pkl.
        results_path_extension (str, optional): extension to add to the results path.
            Defaults to None, which will use the default path.
        
    Returns:
        Dict: simulation results
            - params: parameters of the simulation
                - num_segments: number of segments
                - type_of_derivation: type of derivation
                - type_of_integration: type of integration
                - param_integration: parameter for the integration
                - robot_params: parameters of the robot
                - strain_selector: selector for the strains
                - q0: initial configuration
                - q_d0: initial velocity
                - t: time of simulation
                - dt: time step
                - video_ts: time steps for video
            - results: results of the simulation
                - q_ts: configuration trajectory
                - q_d_ts: velocity trajectory
                - chi_ee_ts: end-effector position trajectory
                - U_ts: potential energy trajectory
                - T_ts: kinetic energy trajectory
            - execution_time: execution time of each step
                - import_model: time to import the model
                - compile_dynamical_matrices: time to JIT-compile the dynamical matrices function
                - evaluate_dynamical_matrices: time to evaluate the dynamical matrices function
                - compile_ode: time to JIT-compile the ODE function
                - evaluate_ode: time to evaluate the ODE function
                - evaluate_forward_kinematics: time to evaluate the forward kinematics function
                - evaluate_potential_energy: time to evaluate the potential energy function
                - evaluate_kinetic_energy: time to evaluate the kinetic energy function
                - draw_robot: time to draw the robot
    """

    # ===================================================
    # Initialization of the simulation parameters 
    # ===================================================
    
    if not isinstance(num_segments, int):
        raise TypeError(f"num_segments must be an integer, but got {type(num_segments).__name__}")
    if num_segments < 1:
        raise ValueError(f"num_segments must be greater than 0, but got {num_segments}")
    
    if not isinstance(type_of_derivation, str):
        raise TypeError(f"type_of_derivation must be a string, but got {type(type_of_derivation).__name__}")
    if type_of_derivation == "numeric":
        if not isinstance(type_of_integration, str):
            raise TypeError(f"type_of_integration must be a string, but got {type(type_of_integration).__name__}")
        if param_integration is None:
            if type_of_integration == "gauss-legendre":
                param_integration = 5
            if type_of_integration == "gauss-kronrad":
                param_integration = 15
            elif type_of_integration == "trapezoid":
                param_integration = 1000
        if not isinstance(param_integration, int):
            raise TypeError(f"param_integration must be an integer, but got {type(param_integration).__name__}")
        if param_integration < 1:
            raise ValueError(f"param_integration must be greater than 0, but got {param_integration}")
        
        if type_of_jacobian not in ["explicit", "autodiff"]:
            raise ValueError(
                f"type_of_jacobian must be 'explicit' or 'autodiff', but got {type_of_jacobian}"
            )    
                
    elif type_of_derivation == "symbolic":
        # filepath to symbolic expressions
        sym_exp_filepath = (
            Path(jsrm.__file__).parent
            / "symbolic_expressions"
            / f"planar_pcs_ns-{num_segments}.dill"
        )
        if not sym_exp_filepath.exists():
            raise FileNotFoundError(
                f"File {sym_exp_filepath} does not exist. Please run the script to generate the symbolic expressions."
            )
    else:
        raise ValueError(
            f"type_of_derivation must be 'symbolic' or 'numeric', but got {type_of_derivation}"
        )

    if robot_params is None:
        # set parameters
        rho = 1070 * jnp.ones((num_segments,))      # Volumetric density of Dragon Skin 20 [kg/m^3]
        robot_params = {
            "th0": jnp.array(0.0),                  # Initial orientation angle [rad]
            "l": 1e-1 * jnp.ones((num_segments,)),  # Length of each segment [m]
            "r": 2e-2 * jnp.ones((num_segments,)),  # Radius of each segment [m]
            "rho": rho,
            "g": jnp.array([0.0, 9.81]),            # Gravity vector [m/s^2]
            "E": 2e3 * jnp.ones((num_segments,)),   # Elastic modulus [Pa]
            "G": 1e3 * jnp.ones((num_segments,)),   # Shear modulus [Pa]
        }
        robot_params["D"] = 1e-3 * jnp.diag(              # Damping matrix [Ns/m]
            (jnp.repeat(
                jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0
            ) * robot_params["l"][:, None]).flatten()
        )
    #TODO: check if the parameters are correctly defined
    
    # Max number of degrees of freedom = size of the strain vector
    n_xi = 3 * num_segments

    if strain_selector is None:
        # activate all strains (i.e. bending, shear, and axial)
        strain_selector = jnp.ones((n_xi,), dtype=bool)
    if not isinstance(strain_selector, jnp.ndarray):
        if isinstance(strain_selector, list):
            strain_selector = jnp.array(strain_selector)
        else:
            raise TypeError(f"strain_selector must be a jnp.ndarray, but got {type(strain_selector).__name__}")
    strain_selector = strain_selector.flatten()
    if strain_selector.shape[0] != n_xi:
        raise ValueError(
            f"strain_selector must have the same shape as the strain vector, but got {strain_selector.shape[0]} instead of {n_xi}"
        )
    if not jnp.issubdtype(strain_selector.dtype, jnp.bool_):
        raise TypeError(
            f"strain_selector must be a boolean array, but got {strain_selector.dtype}"
        )
        
    # number of generalized coordinates
    n_dof = jnp.sum(strain_selector)
    
    if q0 is None:
        q0_init = jnp.repeat(jnp.array([5.0 * jnp.pi, 0.1, 0.2])[None, :], num_segments, axis=0).flatten()
        q0 = jnp.array([q0_init[i] for i in range(3*num_segments) if strain_selector[i]])
    if not isinstance(q0, jnp.ndarray):
        if isinstance(q0, list):
            q0 = jnp.array(q0)
        else:
            raise TypeError(f"q0 must be a jnp.ndarray, but got {type(q0).__name__}")
    if q0.shape[0] != n_dof:
        raise ValueError(
            f"q0 must have the same shape as jnp.sum(strain_selector), but got {q0.shape[0]} instead of {jnp.sum(strain_selector)}"
        )
    
    if q_d0 is None:
        # define initial velocity
        q_d0 = jnp.zeros_like(q0)
    if not isinstance(q_d0, jnp.ndarray):
        if isinstance(q_d0, list):
            q_d0 = jnp.array(q_d0)
        else:
            raise TypeError(f"q_d0 must be a jnp.ndarray, but got {type(q_d0).__name__}")
    if q_d0.shape[0] != n_dof:
        raise ValueError(
            f"q_d0 must have the same shape as q0, but got {q_d0.shape[0]} instead of {n_dof}"
            )
    
    if not isinstance(t, float):
        if isinstance(t, int):
            t = float(t)
        else:
            raise TypeError(f"t must be a float, but got {type(t).__name__}")
    if t <= 0:
        raise ValueError(f"t must be greater than 0, but got {t}")
    
    if dt is None:
        dt = 1e-4
    if not isinstance(dt, float):
        if isinstance(dt, int):
            dt = float(dt)
        else:
            raise TypeError(f"dt must be a float, but got {type(dt).__name__}")
    if dt <= 0:
        raise ValueError(f"dt must be greater than 0, but got {dt}")
    if dt > t:
        raise ValueError(f"dt must be less than t, but got {dt} > {t}")
    
    if not isinstance(bool_print, bool):
        raise TypeError(f"bool_print must be a boolean, but got {type(bool_print).__name__}")
    if not isinstance(bool_plot, bool):
        raise TypeError(f"bool_plot must be a boolean, but got {type(bool_plot).__name__}")
    if not isinstance(bool_save_plot, bool):
        raise TypeError(f"bool_save_plot must be a boolean, but got {type(bool_save_plot).__name__}")
    if not isinstance(bool_save_video, bool):
        raise TypeError(f"bool_save_video must be a boolean, but got {type(bool_save_video).__name__}")
    if not isinstance(bool_save_res, bool):
        raise TypeError(f"bool_save_res must be a boolean, but got {type(bool_save_res).__name__}")
    
    if bool_save_res:
        if results_path is None:
            results_path_parent = (
                Path(__file__).parent 
                / "results"
                / "planar_pcs_results"
                / f"ns-{num_segments}")
            file_name = f"{('symb' if type_of_derivation == 'symbolic' else 'num')}"
            if type_of_derivation == "numeric":
                file_name += f"-{type_of_integration}-{param_integration}-{type_of_jacobian}"

            if results_path_extension is not None:
                if not isinstance(results_path_extension, str):
                    raise TypeError(
                        f"results_path_extension must be a string, but got {type(results_path_extension).__name__}"
                    )
                file_name += f"-{results_path_extension}"

            results_path = (results_path_parent / file_name).with_suffix(".pkl")
        
        if isinstance(results_path, str) or isinstance(results_path, Path):
            results_path = Path(results_path)
            if results_path.suffix != ".pkl":
                raise ValueError(
                    f"results_path must have the suffix .pkl, but got {results_path.suffix}"
                )
            else:
                results_path = Path(results_path)
        else:
            raise TypeError(
                f"results_path must be a string, but got {type(results_path).__name__}"
            )
        
        # create the directory if it does not exist     
        if not results_path.parent.exists():
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
    # =============================================================================================
    # Figures and video generation
    # =============================================================================================
    extension = f"planar_pcs_ns-{num_segments}-{('symb' if type_of_derivation == 'symbolic' else 'num')}"
    if type_of_derivation == "numeric":
        extension += f"-{type_of_integration}-{param_integration}"

    # ======================
    # For video generation
    # ======================
    # set simulation parameters
    ts = jnp.arange(0.0, t, dt)  # time steps
    skip_step = 10  # how many time steps to skip in between video frames
    video_ts = ts[::skip_step]  # time steps for video

    # video settings
    video_width, video_height = 700, 700  # img height and width"
    video_path_parent = Path(__file__).parent / "videos" / "planar_pcs" / f"ns-{num_segments}"
    if bool_save_video:
        video_path_parent.mkdir(parents=True, exist_ok=True)
    
    video_path = video_path_parent / f"{extension}.mp4"

    def draw_robot(
        batched_forward_kinematics_fn: Callable,
        params: Dict[str, Array],
        q: Array,
        width: int,
        height: int,
        num_points: int = 50,
    ) -> onp.ndarray:
        """
        Draw the robot in an array of shape (width, height, 3) in OpenCV format.
        The robot is drawn as a curve in the x-y plane. The base is drawn as a rectangle.

        Args:
            batched_forward_kinematics_fn (Callable): function to compute the forward kinematics.
            params (Dict[str, Array]):  parameters of the robot.
            q (Array): configuration of the robot.
            width (int): width of the image.
            height (int): height of the image.
            num_points (int, optional):  number of points to draw the robot.
                Defaults to 50.

        Returns:
            onp.ndarray: image of the robot in OpenCV format.
        """
        # plotting in OpenCV
        h, w = height, width  # img height and width
        ppm = h / (2.0 * jnp.sum(params["l"]))  # pixel per meter
        base_color = (0, 0, 0)  # black robot_color in BGR
        robot_color = (255, 0, 0)  # black robot_color in BGR

        # we use for plotting N points along the length of the robot
        s_ps = jnp.linspace(0, jnp.sum(params["l"]), num_points)

        # poses along the robot of shape (3, N)
        chi_ps = batched_forward_kinematics_fn(params, q, s_ps)

        img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
        curve_origin = onp.array(
            [w // 2, 0.1 * h], dtype=onp.int32
        )  # in x-y pixel coordinates
        # draw base
        cv2.rectangle(img, (0, h - curve_origin[1]), (w, h), color=base_color, thickness=-1)
        # transform robot poses to pixel coordinates
        # should be of shape (N, 2)
        curve = onp.array((curve_origin + chi_ps[:2, :].T * ppm), dtype=onp.int32)
        # invert the v pixel coordinate
        curve[:, 1] = h - curve[:, 1]
        cv2.polylines(img, [curve], isClosed=False, color=robot_color, thickness=10)

        return img

    # ======================
    # For figure saving
    # ======================
    figures_path_parent = Path(__file__).parent / "figures" / "planar_pcs" / f"ns-{num_segments}"
    figures_path_parent.mkdir(parents=True, exist_ok=True) 

    # =====================================================================================================
    # Simulation
    # =====================================================================================================
    # save the simulation parameters and results
    simulation_dict = {
        "params": {
            "num_segments": num_segments,
            "type_of_derivation": type_of_derivation,
            "type_of_integration": type_of_integration,
            "param_integration": param_integration,
            "type_of_jacobian": type_of_jacobian,
            "robot_params": robot_params,
            "strain_selector": strain_selector,
            "q0": q0,
            "q_d0": q_d0,
            "t": t,
            "dt": dt,
            "video_ts": video_ts
        },
        "results": {
        },
        "execution_time": {
        }
    }
    if bool_save_video:
        simulation_dict["execution_time"]["draw_robot"] = None
        

    print("Number of segments:", num_segments)
    print("Type of derivation:", type_of_derivation)
    if type_of_derivation == "numeric":
        print("Type of integration:", type_of_integration)
        print("Parameter for integration:", param_integration)
    print()
    
    # ====================================================
    # Import the planar PCS model depending on the type of derivation
    print("Importing the planar PCS model...")
    
    timer_start = time.time()
    if type_of_derivation == "symbolic":
        strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
            planar_pcs.factory(sym_exp_filepath, strain_selector)
        )
    elif type_of_derivation == "numeric":
        strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
            planar_pcs_num.factory(
                num_segments, 
                strain_selector, 
                integration_type=type_of_integration, 
                param_integration=param_integration, 
                jacobian_type=type_of_jacobian
                )
        )
    timer_end = time.time()
    
    simulation_dict["execution_time"]["import_model"] = timer_end - timer_start
    
    print(f"Importing the planar PCS model took {timer_end - timer_start:.2e} seconds. \n")
    
    # ====================================================  
    # JIT the functions
    print("JIT-compiling the dynamical matrices function...")
    dynamical_matrices_fn = jax.jit(partial(dynamical_matrices_fn))
    
    # First evaluation of the dynamical matrices to trigger JIT compilation
    print(f"Evaluating the dynamical matrices for the first time (JIT-compilation) for q0 = {q0} and q_d0 = {jnp.zeros_like(q0)}...")
    
    timer_start = time.time()
    B, C, G, K, D, A = dynamical_matrices_fn(robot_params, q0, jnp.zeros_like(q0))
    B.block_until_ready()  # ensure the matrices are computed
    C.block_until_ready()  # ensure the matrices are computed
    G.block_until_ready()  # ensure the matrices are computed
    K.block_until_ready()  # ensure the matrices are computed
    D.block_until_ready()  # ensure the matrices are computed
    A.block_until_ready()  # ensure the matrices are computed
    timer_end = time.time()
    
    simulation_dict["execution_time"]["compile_dynamical_matrices"] = timer_end - timer_start
    
    if bool_print:
        print("B =\n", B) 
        print("C =\n", C)
        print("G =\n", G)
        print("K =\n", K)
        print("D =\n", D)
        print("A =\n", A)
    
    print(f"Evaluating the dynamical matrices for the first time took {timer_end - timer_start:.2e} seconds. \n")

    # Second evaluation of the dynamical matrices to capture the time of the evaluation
    print("Evaluating the dynamical matrices for the second time (JIT-evaluation)...")
    
    timer_start = time.time()
    B, C, G, K, D, A = dynamical_matrices_fn(robot_params, q0, jnp.zeros_like(q0))
    B.block_until_ready()  # ensure the matrices are computed
    C.block_until_ready()  # ensure the matrices are computed
    G.block_until_ready()  # ensure the matrices are computed
    K.block_until_ready()  # ensure the matrices are computed
    D.block_until_ready()  # ensure the matrices are computed
    A.block_until_ready()  # ensure the matrices are computed
    timer_end = time.time()
    
    simulation_dict["execution_time"]["evaluate_dynamical_matrices"] = timer_end - timer_start
    
    simulation_dict["results"]["B_q0"] = B
    simulation_dict["results"]["C_q0"] = C
    simulation_dict["results"]["G_q0"] = G
    simulation_dict["results"]["K_q0"] = K
    simulation_dict["results"]["D_q0"] = D
    simulation_dict["results"]["A_q0"] = A
    
    print(f"Evaluating the dynamical matrices for the second time took {timer_end - timer_start:.2e} seconds. \n")
    
    # ====================================================
    # Parameter for the simulation
    x0 = jnp.concatenate([q0, q_d0])  # initial condition
    tau = jnp.zeros_like(q0)  # torques

    # Create the ODE function
    ode_fn = ode_factory(dynamical_matrices_fn, robot_params, tau)
    term = ODETerm(ode_fn)

    # JIT the functions
    print("JIT-compiling the ODE function...")
    diffeqsolve_fn = jax.jit(
        partial(diffeqsolve, 
                term, 
                solver=Tsit5(), 
                t0=ts[0], 
                t1=ts[-1], 
                dt0=dt, 
                y0=x0, 
                max_steps=None, 
                saveat=SaveAt(ts=video_ts))
        )
    
    # ====================================================
    # First evaluation of the ODE to trigger JIT compilation
    print("Solving the ODE for the first time (JIT-compilation)...")
    
    timer_start = time.time()
    sol = diffeqsolve_fn()
    sol.ys.block_until_ready()  # ensure the solution is computed
    timer_end = time.time()
        
    simulation_dict["execution_time"]["compile_ode"] = timer_end - timer_start
    
    print(f"Solving the ODE for the first time took {timer_end - timer_start:.2e} seconds. \n")
    
    # Second evaluation of the ODE to capture the time of the evaluation
    print("Solving the ODE for the second time (JIT-evaluation)...")
    
    timer_start = time.time()
    sol = diffeqsolve_fn()
    sol.ys.block_until_ready()  # ensure the solution is computed
    timer_end = time.time()
    
    simulation_dict["execution_time"]["evaluate_ode"] = timer_end - timer_start
    
    # the evolution of the generalized coordinates
    q_ts = sol.ys[:, :n_dof].block_until_ready()
    
    # the evolution of the generalized velocities
    q_d_ts = sol.ys[:, n_dof:].block_until_ready()
    
    if bool_print:
        print("q_ts =\n", q_ts)
        print("q_d_ts =\n", q_d_ts)
        
    simulation_dict["results"]["q_ts"] = q_ts
    simulation_dict["results"]["q_d_ts"] = q_d_ts
    
    print(f"Solving the ODE for a second time took {timer_end - timer_start:.2e} seconds. \n")

    # ====================================================
    # First evaluation of the forward kinematics to trigger JIT compilation
    print("Evaluating the forward kinematics of the end-effector along the trajectory for the first time (JIT-compilation)...")
    
    timer_start = time.time()
    chi_ee_ts = vmap(forward_kinematics_fn, in_axes=(None, 0, None))(
        robot_params, q_ts, jnp.array([jnp.sum(robot_params["l"])])
    ).block_until_ready()
    timer_end = time.time()
    
    simulation_dict["execution_time"]["compile_forward_kinematics"] = timer_end - timer_start
    
    if bool_print:
        print("chi_ee_ts =\n", chi_ee_ts)
    
    print(f"Evaluating the forward kinematics for the first time took {timer_end - timer_start:.2e} seconds. \n")
    
    # Second evaluation of the forward kinematics to capture the time of the evaluation
    print("Evaluating the forward kinematics of the end-effector along the trajectory for a second time (JIT-evaluation)...")
    
    timer_start = time.time()
    chi_ee_ts = vmap(forward_kinematics_fn, in_axes=(None, 0, None))(
        robot_params, q_ts, jnp.array([jnp.sum(robot_params["l"])])
    ).block_until_ready()
    timer_end = time.time()
    
    simulation_dict["execution_time"]["evaluate_forward_kinematics"] = timer_end - timer_start
    
    simulation_dict["results"]["chi_ee_ts"] = chi_ee_ts
    
    print(f"Evaluating the forward kinematics for a second time took {timer_end - timer_start:.2e} seconds. \n")
    
    #====================================================
    # Plotting
    #===================
    if bool_plot or bool_save_plot:
        print("Plotting the results... \n")
        
        
        # Plot the configuration vs time
        plt.figure()
        plt.title("Configuration vs Time")
        for segment_idx in range(num_segments):
            plt.plot(
                video_ts, q_ts[:, 3 * segment_idx + 0],
                label=r"$\kappa_\mathrm{be," + str(segment_idx + 1) + "}$ [rad/m]"
            )
            plt.plot(
                video_ts, q_ts[:, 3 * segment_idx + 1],
                label=r"$\sigma_\mathrm{sh," + str(segment_idx + 1) + "}$ [-]"
            )
            plt.plot(
                video_ts, q_ts[:, 3 * segment_idx + 2],
                label=r"$\sigma_\mathrm{ax," + str(segment_idx + 1) + "}$ [-]"
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Configuration")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if bool_save_plot:
            plt.savefig(
                figures_path_parent / f"config_vs_time_{extension}.png", bbox_inches="tight", dpi=300
            )
            print("Figures saved at", figures_path_parent / f"config_vs_time_{extension}.png")
        if bool_plot:
            plt.show()
        plt.close()
        
        # Plot end-effector position vs time
        plt.figure()
        plt.title("End-effector position vs Time")
        plt.plot(video_ts, chi_ee_ts[:, 0], label="x")
        plt.plot(video_ts, chi_ee_ts[:, 1], label="y")
        plt.xlabel("Time [s]")
        plt.ylabel("End-effector Position [m]")
        plt.legend()
        plt.grid(True)
        plt.box(True)
        plt.tight_layout()
        if bool_save_plot:
            plt.savefig(
                figures_path_parent / f"end_effector_position_vs_time_{extension}.png", bbox_inches="tight", dpi=300
            )
            print("Figures saved at", figures_path_parent / f"end_effector_position_vs_time_{extension}.png ")
        if bool_plot:
            plt.show()
        plt.close()
        
        # plot the end-effector position in the x-y plane as a scatter plot with the time as the color
        plt.figure()
        plt.title("End-effector position in the x-y plane")
        plt.scatter(chi_ee_ts[:, 0], chi_ee_ts[:, 1], c=video_ts, cmap="viridis")
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("End-effector x [m]")
        plt.ylabel("End-effector y [m]")
        plt.colorbar(label="Time [s]")
        plt.tight_layout()
        if bool_save_plot:
            plt.savefig(
                figures_path_parent / f"end_effector_position_xy_{extension}.png", bbox_inches="tight", dpi=300
            )
            print("Figures saved at", figures_path_parent / f"end_effector_position_xy_{extension}.png \n")
        if bool_plot:
            plt.show()
        plt.close()
    
    # ====================================================
    # First evaluation of the potential energy to trigger JIT compilation
    print("Evaluating the potential energy for the first time (JIT-compilation)...")
    
    timer_start = time.time()
    U_ts = vmap(partial(auxiliary_fns["potential_energy_fn"], robot_params))(q_ts).block_until_ready()
    U_ts = U_ts.block_until_ready()  # ensure the potential energy is computed
    timer_end = time.time()
    
    simulation_dict["execution_time"]["compile_potential_energy"] = timer_end - timer_start
    
    print(f"Evaluating the potential energy took {timer_end - timer_start:.2e} seconds. \n")
    
    # Second evaluation of the potential energy to capture the time of the evaluation
    print("Evaluating the potential energy for a second time (JIT-evaluation)...")
    
    timer_start = time.time()
    U_ts = vmap(partial(auxiliary_fns["potential_energy_fn"], robot_params))(q_ts).block_until_ready()
    U_ts = U_ts.block_until_ready()  # ensure the potential energy is computed
    timer_end = time.time()
    
    simulation_dict["execution_time"]["evaluate_potential_energy"] = timer_end - timer_start
    
    simulation_dict["results"]["U_ts"] = U_ts
    
    print(f"Evaluating the potential energy for a second time took {timer_end - timer_start:.2e} seconds. \n")
    
    # ====================================================
    # First evaluation of the kinetic energy to trigger JIT compilation
    print("Evaluating the kinetic energy for the first time (JIT-compilation)...")
    
    timer_start = time.time()
    T_ts = vmap(partial(auxiliary_fns["kinetic_energy_fn"], robot_params))(q_ts, q_d_ts).block_until_ready()
    T_ts = T_ts.block_until_ready()  # ensure the kinetic energy is computed
    timer_end = time.time()
    
    simulation_dict["execution_time"]["compile_kinetic_energy"] = timer_end - timer_start
    
    print(f"Evaluating the kinetic energy took {timer_end - timer_start:.2e} seconds. \n")
    
    # Second evaluation of the kinetic energy to capture the time of the evaluation
    print("Evaluating the kinetic energy for a second time (JIT-evaluation)...")
      
    timer_start = time.time()
    T_ts = vmap(partial(auxiliary_fns["kinetic_energy_fn"], robot_params))(q_ts, q_d_ts).block_until_ready()
    T_ts = T_ts.block_until_ready()  # ensure the kinetic energy is computed
    timer_end = time.time()
    
    simulation_dict["execution_time"]["evaluate_kinetic_energy"] = timer_end - timer_start
    
    simulation_dict["results"]["T_ts"] = T_ts
    
    print(f"Evaluating the kinetic energy for a second time took {timer_end - timer_start:.2e} seconds. \n")
    
    if bool_print:
        print("U_ts =\n", U_ts)
        print("T_ts =\n", T_ts)
    
    if bool_plot or bool_save_plot:
        print("Plotting the energy... \n")
        # Plot the energy vs time
        plt.figure()
        plt.title("Energy vs Time")
        plt.plot(video_ts, U_ts, label="Potential energy")
        plt.plot(video_ts, T_ts, label="Kinetic energy")
        plt.xlabel("Time [s]")
        plt.ylabel("Energy [J]")
        plt.legend()
        plt.grid(True)
        plt.box(True)
        plt.tight_layout()
        if bool_save_plot:
            plt.savefig(
                figures_path_parent / f"energy_vs_time_{extension}.png", bbox_inches="tight", dpi=300
            )
            print("Figures saved at", figures_path_parent / f"energy_vs_time_{extension}.png \n")
        if bool_plot:
            plt.show()
        plt.close()

    # ====================================================
    # Video generation
    # =================
    if bool_save_video:
        print("Drawing the robot...")
        
        # Vectorize the forward kinematics function according to the s coordinates
        print("Vectorizing the forward kinematics function according to the number of segments...")
        batched_forward_kinematics = vmap(
            forward_kinematics_fn, in_axes=(None, None, 0), out_axes=-1
        )
        
        # Initialize the video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            str(video_path),
            fourcc,
            1 / (skip_step * dt),  # fps
            (video_width, video_height),
        )

        # Create the video frames
        for time_idx, t in enumerate(video_ts):
            x = sol.ys[time_idx]
            img = draw_robot(
                batched_forward_kinematics,
                robot_params,
                x[: (x.shape[0] // 2)],
                video_width,
                video_height,
            )
            video.write(img)

        # Release the video
        video.release()
        
        print(f"Video saved at {video_path}. \n")
    
    # ===========================================================================
    # Save the simulation results
    # ===========================
    if bool_save_res:
        print("Saving the simulation results...")
        with open(results_path, "wb") as f:
            pickle.dump(simulation_dict, f)
        print(f"Simulation results saved at {results_path} \n")
    else:
        print("Simulation results not saved. \n")
        
    print("Simulation finished. \n")
    print("**************************************************************** \n \n")
        
    return simulation_dict

if __name__ == "__main__":
    # Example of usage
    simulate_res = simulate_planar_pcs(
        num_segments        = 1,
        type_of_derivation  = "numeric", #"symbolic"
        type_of_integration = "gauss", # "trapezoid"
        # param_integration   = 30,
        strain_selector     = [True, False, True],
        bool_print          = True,
        bool_plot           = True,
        bool_save_plot      = False, 
        bool_save_video     = False,
        bool_save_res       = False
    )
    
def simulate_planar_pcs_time_eval(
    num_segments: int,
    type_of_derivation: str = "symbolic",
    type_of_integration: str = "gauss-legendre",
    param_integration: int = None,
    type_of_jacobian: str = "explicit",
    robot_params: Dict[str, Array] = None,
    strain_selector: Array = None,
    q0: Array = None,
    q_d0: Array = None,
    t: float = 1.0,
    dt: float = None,
    bool_save_res: bool = False,
    results_path: str = None, 
    results_path_extension: str = None,
    type_time = "once",
    nb_eval : int = 1000,
    nb_samples: int = 10
) -> Dict:
    """
    Simulate a planar PCS model. Save the video and figures.

    Args:
        num_segments (int): number of segments of the robot.
        type_of_derivation (str, optional): type of derivation ("symbolic" or "numeric"). 
            Defaults to "symbolic".
        type_of_integration (str, optional): scheme of integration ("gauss-legendre", "gauss-kronrad" or "trapezoid"). 
            Defaults to "gauss-legendre".
        param_integration (int, optional): parameter for the integration scheme (number of points for gauss or number of points for trapezoid). 
            Defaults to 1000 for trapezoid and 5 for gauss.
        type_of_jacobian (str, optional): type of jacobian ("explicit" or "autodiff").
            Defaults to "explicit".
        strain_selector (Array, optional): selector for the strains as a boolean array.
            Defaults to all.
        robot_params (Dict[str, Array], optional): parameters of the robot. 
        strain_selector (Array, optional): selector for the strains.
        t (float, optional): time of simulation [s]. 
            Defaults to 1.0s.
        dt (float, optional): time step [s]. 
            Defaults to 1e-4s.
        q0 (Array, optional): initial configuration.
            Defaults to a configuration with 5.0pi rad of bending, 0.1 of shear, and 0.2 of axial strain for each segment.
        q_d0 (Array, optional): initial velocity.
            Defaults to 0 array.
        bool_print (bool, optional): if True, print the simulation results.
            Defaults to True.
        bool_plot (bool, optional): if True, show the figures. 
            Defaults to True.
        bool_save_plot (bool, optional): if True, save the figures.
            Defaults to False.
        bool_save_video (bool, optional): if True, save the video.
            Defaults to False.
        bool_save_res (bool, optional): if True, save the results of the simulation. 
            Defaults to False.
        results_path (str, optional): path to save the dictionary with the simulation results. Must have the suffix .pkl.
        results_path_extension (str, optional): extension to add to the results path.
            Defaults to None, which will use the default path.
        
    Returns:
        Dict: simulation results
            - params: parameters of the simulation
                - num_segments: number of segments
                - type_of_derivation: type of derivation
                - type_of_integration: type of integration
                - param_integration: parameter for the integration
                - robot_params: parameters of the robot
                - strain_selector: selector for the strains
                - q0: initial configuration
                - q_d0: initial velocity
                - t: time of simulation
                - dt: time step
                - video_ts: time steps for video
            - results: results of the simulation
                - q_ts: configuration trajectory
                - q_d_ts: velocity trajectory
                - chi_ee_ts: end-effector position trajectory
                - U_ts: potential energy trajectory
                - T_ts: kinetic energy trajectory
            - execution_time: execution time of each step
                - import_model: time to import the model
                - compile_dynamical_matrices: time to JIT-compile the dynamical matrices function
                - evaluate_dynamical_matrices: time to evaluate the dynamical matrices function
                - compile_ode: time to JIT-compile the ODE function
                - evaluate_ode: time to evaluate the ODE function
                - evaluate_forward_kinematics: time to evaluate the forward kinematics function
                - evaluate_potential_energy: time to evaluate the potential energy function
                - evaluate_kinetic_energy: time to evaluate the kinetic energy function
                - draw_robot: time to draw the robot
    """

    # ===================================================
    # Initialization of the simulation parameters 
    # ===================================================
    
    if not isinstance(num_segments, int):
        raise TypeError(f"num_segments must be an integer, but got {type(num_segments).__name__}")
    if num_segments < 1:
        raise ValueError(f"num_segments must be greater than 0, but got {num_segments}")
    
    if not isinstance(type_of_derivation, str):
        raise TypeError(f"type_of_derivation must be a string, but got {type(type_of_derivation).__name__}")
    if type_of_derivation == "numeric":
        if not isinstance(type_of_integration, str):
            raise TypeError(f"type_of_integration must be a string, but got {type(type_of_integration).__name__}")
        if param_integration is None:
            if type_of_integration == "gauss-legendre":
                param_integration = 5
            if type_of_integration == "gauss-kronrad":
                param_integration = 15
            elif type_of_integration == "trapezoid":
                param_integration = 1000
        if not isinstance(param_integration, int):
            raise TypeError(f"param_integration must be an integer, but got {type(param_integration).__name__}")
        if param_integration < 1:
            raise ValueError(f"param_integration must be greater than 0, but got {param_integration}")
        
        if type_of_jacobian not in ["explicit", "autodiff"]:
            raise ValueError(
                f"type_of_jacobian must be 'explicit' or 'autodiff', but got {type_of_jacobian}"
            )    
                
    elif type_of_derivation == "symbolic":
        # filepath to symbolic expressions
        sym_exp_filepath = (
            Path(jsrm.__file__).parent
            / "symbolic_expressions"
            / f"planar_pcs_ns-{num_segments}.dill"
        )
        if not sym_exp_filepath.exists():
            raise FileNotFoundError(
                f"File {sym_exp_filepath} does not exist. Please run the script to generate the symbolic expressions."
            )
    else:
        raise ValueError(
            f"type_of_derivation must be 'symbolic' or 'numeric', but got {type_of_derivation}"
        )

    if robot_params is None:
        # set parameters
        rho = 1070 * jnp.ones((num_segments,))      # Volumetric density of Dragon Skin 20 [kg/m^3]
        robot_params = {
            "th0": jnp.array(0.0),                  # Initial orientation angle [rad]
            "l": 1e-1 * jnp.ones((num_segments,)),  # Length of each segment [m]
            "r": 2e-2 * jnp.ones((num_segments,)),  # Radius of each segment [m]
            "rho": rho,
            "g": jnp.array([0.0, 9.81]),            # Gravity vector [m/s^2]
            "E": 2e3 * jnp.ones((num_segments,)),   # Elastic modulus [Pa]
            "G": 1e3 * jnp.ones((num_segments,)),   # Shear modulus [Pa]
        }
        robot_params["D"] = 1e-3 * jnp.diag(              # Damping matrix [Ns/m]
            (jnp.repeat(
                jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0
            ) * robot_params["l"][:, None]).flatten()
        )
    #TODO: check if the parameters are correctly defined
    
    # Max number of degrees of freedom = size of the strain vector
    n_xi = 3 * num_segments

    if strain_selector is None:
        # activate all strains (i.e. bending, shear, and axial)
        strain_selector = jnp.ones((n_xi,), dtype=bool)
    if not isinstance(strain_selector, jnp.ndarray):
        if isinstance(strain_selector, list):
            strain_selector = jnp.array(strain_selector)
        else:
            raise TypeError(f"strain_selector must be a jnp.ndarray, but got {type(strain_selector).__name__}")
    strain_selector = strain_selector.flatten()
    if strain_selector.shape[0] != n_xi:
        raise ValueError(
            f"strain_selector must have the same shape as the strain vector, but got {strain_selector.shape[0]} instead of {n_xi}"
        )
    if not jnp.issubdtype(strain_selector.dtype, jnp.bool_):
        raise TypeError(
            f"strain_selector must be a boolean array, but got {strain_selector.dtype}"
        )
        
    # number of generalized coordinates
    n_dof = jnp.sum(strain_selector)
    
    if q0 is None:
        q0_init = jnp.repeat(jnp.array([5.0 * jnp.pi, 0.1, 0.2])[None, :], num_segments, axis=0).flatten()
        q0 = jnp.array([q0_init[i] for i in range(3*num_segments) if strain_selector[i]])
    if not isinstance(q0, jnp.ndarray):
        if isinstance(q0, list):
            q0 = jnp.array(q0)
        else:
            raise TypeError(f"q0 must be a jnp.ndarray, but got {type(q0).__name__}")
    if q0.shape[0] != n_dof:
        raise ValueError(
            f"q0 must have the same shape as jnp.sum(strain_selector), but got {q0.shape[0]} instead of {jnp.sum(strain_selector)}"
        )
    
    if q_d0 is None:
        # define initial velocity
        q_d0 = jnp.zeros_like(q0)
    if not isinstance(q_d0, jnp.ndarray):
        if isinstance(q_d0, list):
            q_d0 = jnp.array(q_d0)
        else:
            raise TypeError(f"q_d0 must be a jnp.ndarray, but got {type(q_d0).__name__}")
    if q_d0.shape[0] != n_dof:
        raise ValueError(
            f"q_d0 must have the same shape as q0, but got {q_d0.shape[0]} instead of {n_dof}"
            )
    
    if not isinstance(t, float):
        if isinstance(t, int):
            t = float(t)
        else:
            raise TypeError(f"t must be a float, but got {type(t).__name__}")
    if t <= 0:
        raise ValueError(f"t must be greater than 0, but got {t}")
    
    if dt is None:
        dt = 1e-4
    if not isinstance(dt, float):
        if isinstance(dt, int):
            dt = float(dt)
        else:
            raise TypeError(f"dt must be a float, but got {type(dt).__name__}")
    if dt <= 0:
        raise ValueError(f"dt must be greater than 0, but got {dt}")
    if dt > t:
        raise ValueError(f"dt must be less than t, but got {dt} > {t}")
    
    if not isinstance(bool_save_res, bool):
        raise TypeError(f"bool_save_res must be a boolean, but got {type(bool_save_res).__name__}")
    
    if bool_save_res:
        if results_path is None:
            results_path_parent = (
                Path(__file__).parent 
                / "results"
                / "planar_pcs_results"
                / f"ns-{num_segments}")
            file_name = f"{('symb' if type_of_derivation == 'symbolic' else 'num')}"
            if type_of_derivation == "numeric":
                file_name += f"-{type_of_integration}-{param_integration}-{type_of_jacobian}"

            file_name += f"-{type_time}-{nb_eval}-{nb_samples}"

            if results_path_extension is not None:
                if not isinstance(results_path_extension, str):
                    raise TypeError(
                        f"results_path_extension must be a string, but got {type(results_path_extension).__name__}"
                    )
                file_name += f"-{results_path_extension}"

            results_path = (results_path_parent / file_name).with_suffix(".pkl")
        
        if isinstance(results_path, str) or isinstance(results_path, Path):
            results_path = Path(results_path)
            if results_path.suffix != ".pkl":
                raise ValueError(
                    f"results_path must have the suffix .pkl, but got {results_path.suffix}"
                )
            else:
                results_path = Path(results_path)
        else:
            raise TypeError(
                f"results_path must be a string, but got {type(results_path).__name__}"
            )
        
        # create the directory if it does not exist     
        if not results_path.parent.exists():
            results_path.parent.mkdir(parents=True, exist_ok=True)

    # set simulation parameters
    ts = jnp.arange(0.0, t, dt)  # time steps
    skip_step = 10  # how many time steps to skip in between video frames
    video_ts = ts[::skip_step]  # time steps for video

    # =====================================================================================================
    # Simulation
    # =====================================================================================================
    # save the simulation parameters and results
    simulation_dict = {
        "params": {
            "num_segments": num_segments,
            "type_of_derivation": type_of_derivation,
            "type_of_integration": type_of_integration,
            "param_integration": param_integration,
            "type_of_jacobian": type_of_jacobian,
            "robot_params": robot_params,
            "strain_selector": strain_selector,
            "q0": q0,
            "q_d0": q_d0,
            "t": t,
            "dt": dt, 
            "nb_eval": nb_eval,
            "nb_samples": nb_samples,
        },
        "execution_time": {
        }
    }
        

    print("Number of segments:", num_segments)
    print("Type of derivation:", type_of_derivation)
    if type_of_derivation == "numeric":
        print("Type of integration:", type_of_integration)
        print("Parameter for integration:", param_integration)
    print()
    
    # ====================================================
    # Import the planar PCS model depending on the type of derivation
    print("Importing the planar PCS model...")
    
    timer_start = time.time()
    if type_of_derivation == "symbolic":
        strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
            planar_pcs.factory(sym_exp_filepath, strain_selector)
        )
    elif type_of_derivation == "numeric":
        strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
            planar_pcs_num.factory(
                num_segments, 
                strain_selector, 
                integration_type=type_of_integration, 
                param_integration=param_integration, 
                jacobian_type=type_of_jacobian
                )
        )
    timer_end = time.time()
    
    simulation_dict["execution_time"]["import_model"] = timer_end - timer_start
    
    print(f"Importing the planar PCS model took {timer_end - timer_start:.2e} seconds. \n")
    
    if type_time == "once":
        # ====================================================  
        # JIT the functions
        print("JIT-compiling the dynamical matrices function...")
        dynamical_matrices_fn = jax.jit(partial(dynamical_matrices_fn))
        
        # First evaluation of the dynamical matrices to trigger JIT compilation
        print(f"Evaluating the dynamical matrices for the first time (JIT-compilation) for q0 = {q0} and q_d0 = {jnp.zeros_like(q0)}...")
        
        timer_start = time.time()
        B, C, G, K, D, A = dynamical_matrices_fn(robot_params, q0, jnp.zeros_like(q0))
        B.block_until_ready()  # ensure the matrices are computed
        C.block_until_ready()  # ensure the matrices are computed
        G.block_until_ready()  # ensure the matrices are computed
        K.block_until_ready()  # ensure the matrices are computed
        D.block_until_ready()  # ensure the matrices are computed
        A.block_until_ready()  # ensure the matrices are computed
        timer_end = time.time()
        
        simulation_dict["execution_time"]["compile_dynamical_matrices_once"] = timer_end - timer_start
        
        print(f"Evaluating the dynamical matrices for the first time took {timer_end - timer_start:.2e} seconds. \n")

        # Second evaluation of the dynamical matrices to capture the time of the evaluation
        print("Evaluating the dynamical matrices for the second time (JIT-evaluation)...")
        
        def time_dynamical_matrices_once():
            B, C, G, K, D, A = dynamical_matrices_fn(robot_params, q0, jnp.zeros_like(q0))
            B.block_until_ready()  # ensure the matrices are computed
            C.block_until_ready()  # ensure the matrices are computed
            G.block_until_ready()  # ensure the matrices are computed
            K.block_until_ready()  # ensure the matrices are computed
            D.block_until_ready()  # ensure the matrices are computed
            A.block_until_ready()  # ensure the matrices are computed
            return None
        
        results_time_dynamical_matrices_once = [timeit.timeit(time_dynamical_matrices_once, number=nb_eval)/nb_eval for _ in range(nb_samples)]
        mean_time_dynamical_matrices_once = statistics.mean(results_time_dynamical_matrices_once)
        std_time_dynamical_matrices_once = statistics.stdev(results_time_dynamical_matrices_once)
        
        simulation_dict["execution_time"]["evaluate_dynamical_matrices_once"] = (mean_time_dynamical_matrices_once, std_time_dynamical_matrices_once)
        
        print(f"Evaluating the dynamical matrices for the second time took {mean_time_dynamical_matrices_once:.2e} seconds +/- {std_time_dynamical_matrices_once:.2e} seconds (mean +/- std) over {nb_samples} samples. \n")
        
        # ====================================================
        # First evaluation of the forward kinematics to trigger JIT compilation
        print("Evaluating the forward kinematics associated with the initial configuration (JIT-compilation)...")
        timer_start = time.time()
        chi_0 = forward_kinematics_fn(robot_params, q0, jnp.array([jnp.sum(robot_params["l"])])).block_until_ready()  # ensure the forward kinematics is computed
        timer_end = time.time()
        
        simulation_dict["execution_time"]["compile_forward_kinematics_once"] = timer_end - timer_start
        
        print(f"Evaluating the forward kinematics for the first time took {timer_end - timer_start:.2e} seconds. \n")
        
        # Second evaluation of the forward kinematics to capture the time of the evaluation
        print("Evaluating the forward kinematics associated with the initial configuration for a second time (JIT-evaluation)...")
        
        def time_forward_kinematics_once():
            chi_ee = forward_kinematics_fn(robot_params, q0, jnp.array([jnp.sum(robot_params["l"])])).block_until_ready()
            return None
        results_time_forward_kinematics_once = [timeit.timeit(time_forward_kinematics_once, number=nb_eval)/nb_eval for _ in range(nb_samples)]
        mean_time_forward_kinematics_once = statistics.mean(results_time_forward_kinematics_once)
        std_time_forward_kinematics_once = statistics.stdev(results_time_forward_kinematics_once)
        
        simulation_dict["execution_time"]["evaluate_forward_kinematics_once"] = (mean_time_forward_kinematics_once, std_time_forward_kinematics_once)
        
        print(f"Evaluating the forward kinematics for a second time took {mean_time_forward_kinematics_once:.2e} seconds +/- {std_time_forward_kinematics_once:.2e} seconds (mean +/- std) over {nb_samples} samples. \n")
        
        # ====================================================
        # First evaluation of the potential energy to trigger JIT compilation
        print("Evaluating the potential energy for the first time (JIT-compilation)...")
        
        timer_start = time.time()
        U_0 = auxiliary_fns["potential_energy_fn"](robot_params, q0).block_until_ready()
        timer_end = time.time()
        
        simulation_dict["execution_time"]["compile_potential_energy_once"] = timer_end - timer_start
        
        print(f"Evaluating the potential energy for the first time took {timer_end - timer_start:.2e} seconds. \n")
        
        # Second evaluation of the potential energy to capture the time of the evaluation
        print("Evaluating the potential energy for a second time (JIT-evaluation)...")
        
        def time_potential_energy_once():
            U_0 = auxiliary_fns["potential_energy_fn"](robot_params, q0).block_until_ready()
            return None
        results_time_potential_energy_once = [timeit.timeit(time_potential_energy_once, number=nb_eval)/nb_eval for _ in range(nb_samples)]
        mean_time_potential_energy_once = statistics.mean(results_time_potential_energy_once)
        std_time_potential_energy_once = statistics.stdev(results_time_potential_energy_once)
        
        simulation_dict["execution_time"]["evaluate_potential_energy_once"] = (mean_time_potential_energy_once, std_time_potential_energy_once)
        
        print(f"Evaluating the potential energy for a second time took {mean_time_potential_energy_once:.2e} seconds +/- {std_time_potential_energy_once:.2e} seconds (mean +/- std) over {nb_samples} samples. \n")
    
        # ====================================================
        # First evaluation of the kinetic energy to trigger JIT compilation
        print("Evaluating the kinetic energy for the first time (JIT-compilation)...")
        
        timer_start = time.time()
        T_0 = auxiliary_fns["kinetic_energy_fn"](robot_params, q0, jnp.zeros_like(q0)).block_until_ready()
        timer_end = time.time()
        
        simulation_dict["execution_time"]["compile_kinetic_energy_once"] = timer_end - timer_start
        
        print(f"Evaluating the kinetic energy took {timer_end - timer_start:.2e} seconds. \n")
        
        # Second evaluation of the kinetic energy to capture the time of the evaluation
        print("Evaluating the kinetic energy for a second time (JIT-evaluation)...")
        
        def time_kinetic_energy_once():
            T_0 = auxiliary_fns["kinetic_energy_fn"](robot_params, q0, jnp.zeros_like(q0)).block_until_ready()
            return None
        results_time_kinetic_energy_once = [timeit.timeit(time_kinetic_energy_once, number=nb_eval)/nb_eval for _ in range(nb_samples)]
        mean_time_kinetic_energy_once = statistics.mean(results_time_kinetic_energy_once)
        std_time_kinetic_energy_once = statistics.stdev(results_time_kinetic_energy_once)
        
        simulation_dict["execution_time"]["evaluate_kinetic_energy_once"] = (mean_time_kinetic_energy_once, std_time_kinetic_energy_once)
        
        print(f"Evaluating the kinetic energy for a second time took {mean_time_kinetic_energy_once:.2e} seconds +/- {std_time_kinetic_energy_once:.2e} seconds (mean +/- std) over {nb_samples} samples. \n")
    
    else:
        # ====================================================
        # Parameter for the simulation
        x0 = jnp.concatenate([q0, q_d0])  # initial condition
        tau = jnp.zeros_like(q0)  # torques

        # Create the ODE function
        ode_fn = ode_factory(dynamical_matrices_fn, robot_params, tau)
        term = ODETerm(ode_fn)

        # JIT the functions
        print("JIT-compiling the ODE function...")
        diffeqsolve_fn = jax.jit(
            partial(diffeqsolve, 
                    term, 
                    solver=Tsit5(), 
                    t0=ts[0], 
                    t1=ts[-1], 
                    dt0=dt, 
                    y0=x0, 
                    max_steps=None, 
                    saveat=SaveAt(ts=video_ts))
            )
        
        # ====================================================
        # First evaluation of the ODE to trigger JIT compilation
        print("Solving the ODE for the first time (JIT-compilation)...")
        
        timer_start = time.time()
        sol = diffeqsolve_fn()
        sol.ys.block_until_ready()  # ensure the solution is computed
        timer_end = time.time()
            
        simulation_dict["execution_time"]["compile_ode_over_time"] = timer_end - timer_start
        
        print(f"Solving the ODE for the first time took {timer_end - timer_start:.2e} seconds. \n")
        
        # Second evaluation of the ODE to capture the time of the evaluation
        print("Solving the ODE for the second time (JIT-evaluation)...")
        
        def time_diffeqsolve_over_time():
            sol = diffeqsolve_fn()
            sol.ys.block_until_ready()
            return None
        results_time_diffeqsolve_over_time = [timeit.timeit(time_diffeqsolve_over_time, number=nb_eval)/nb_eval for _ in range(nb_samples)]
        mean_time_diffeqsolve_over_time = statistics.mean(results_time_diffeqsolve_over_time)
        std_time_diffeqsolve_over_time = statistics.stdev(results_time_diffeqsolve_over_time)
        
        simulation_dict["execution_time"]["evaluate_ode_over_time"] = (mean_time_diffeqsolve_over_time, std_time_diffeqsolve_over_time)
        
        print(f"Solving the ODE for a second time took {mean_time_diffeqsolve_over_time:.2e} seconds +/- {std_time_diffeqsolve_over_time:.2e} seconds (mean +/- std) over {nb_samples} samples. \n")
        
        # ====================================================
        # the evolution of the generalized coordinates
        q_ts = sol.ys[:, :n_dof].block_until_ready()
        
        # the evolution of the generalized velocities
        q_d_ts = sol.ys[:, n_dof:].block_until_ready()

        # ====================================================
        # First evaluation of the forward kinematics to trigger JIT compilation
        print("Evaluating the forward kinematics of the end-effector along the trajectory for the first time (JIT-compilation)...")
        
        timer_start = time.time()
        chi_ee_ts = vmap(forward_kinematics_fn, in_axes=(None, 0, None))(
            robot_params, q_ts, jnp.array([jnp.sum(robot_params["l"])])
        ).block_until_ready()
        timer_end = time.time()
        
        simulation_dict["execution_time"]["compile_forward_kinematics_over_time"] = timer_end - timer_start
        
        print(f"Evaluating the forward kinematics for the first time took {timer_end - timer_start:.2e} seconds. \n")
        
        # Second evaluation of the forward kinematics to capture the time of the evaluation
        print("Evaluating the forward kinematics of the end-effector along the trajectory for a second time (JIT-evaluation)...")
        
        def time_forward_kinematics_over_time():
            chi_ee_ts = vmap(forward_kinematics_fn, in_axes=(None, 0, None))(
                robot_params, q_ts, jnp.array([jnp.sum(robot_params["l"])])
            ).block_until_ready()
            return None
        results_time_forward_kinematics_over_time = [timeit.timeit(time_forward_kinematics_over_time, number=nb_eval)/nb_eval for _ in range(nb_samples)]
        mean_time_forward_kinematics_over_time = statistics.mean(results_time_forward_kinematics_over_time)
        std_time_forward_kinematics_over_time = statistics.stdev(results_time_forward_kinematics_over_time)
        
        simulation_dict["execution_time"]["evaluate_forward_kinematics_over_time"] = (mean_time_forward_kinematics_over_time, std_time_forward_kinematics_over_time)
        
        print(f"Evaluating the forward kinematics for a second time took {mean_time_forward_kinematics_over_time:.2e} seconds +/- {std_time_forward_kinematics_over_time:.2e} seconds (mean +/- std) over {nb_samples} samples. \n")
    
        # =====================================================
        # First evaluation of the potential energy to trigger JIT compilation
        print("Evaluating the potential energy for the first time (JIT-compilation)...")
        
        timer_start = time.time()
        U_ts = vmap(partial(auxiliary_fns["potential_energy_fn"], robot_params))(q_ts).block_until_ready()
        timer_end = time.time()
        
        simulation_dict["execution_time"]["compile_potential_energy_over_time"] = timer_end - timer_start
        
        print(f"Evaluating the potential energy took {timer_end - timer_start:.2e} seconds. \n")
        
        # Second evaluation of the potential energy to capture the time of the evaluation
        print("Evaluating the potential energy for a second time (JIT-evaluation)...")
        
        def time_potential_energy_over_time():
            U_ts = vmap(partial(auxiliary_fns["potential_energy_fn"], robot_params))(q_ts).block_until_ready()
            return None
        results_time_potential_energy_over_time = [timeit.timeit(time_potential_energy_over_time, number=nb_eval)/nb_eval for _ in range(nb_samples)]
        mean_time_potential_energy_over_time = statistics.mean(results_time_potential_energy_over_time)
        std_time_potential_energy_over_time = statistics.stdev(results_time_potential_energy_over_time)
        
        simulation_dict["execution_time"]["evaluate_potential_energy_over_time"] = (mean_time_potential_energy_over_time, std_time_potential_energy_over_time)
        
        # ====================================================
        # First evaluation of the kinetic energy to trigger JIT compilation
        print("Evaluating the kinetic energy for the first time (JIT-compilation)...")
        
        timer_start = time.time()
        T_ts = vmap(partial(auxiliary_fns["kinetic_energy_fn"], robot_params))(q_ts, q_d_ts).block_until_ready()
        timer_end = time.time()
        
        simulation_dict["execution_time"]["compile_kinetic_energy_over_time"] = timer_end - timer_start
        
        print(f"Evaluating the kinetic energy took {timer_end - timer_start:.2e} seconds. \n")
        
        # Second evaluation of the kinetic energy to capture the time of the evaluation
        print("Evaluating the kinetic energy for a second time (JIT-evaluation)...")
        
        def time_kinetic_energy_over_time():
            T_ts = vmap(partial(auxiliary_fns["kinetic_energy_fn"], robot_params))(q_ts, q_d_ts).block_until_ready()
            return None
        results_time_kinetic_energy_over_time = [timeit.timeit(time_kinetic_energy_over_time, number=nb_eval)/nb_eval for _ in range(nb_samples)]
        mean_time_kinetic_energy_over_time = statistics.mean(results_time_kinetic_energy_over_time)
        std_time_kinetic_energy_over_time = statistics.stdev(results_time_kinetic_energy_over_time)
        
        simulation_dict["execution_time"]["evaluate_kinetic_energy_over_time"] = (mean_time_kinetic_energy_over_time, std_time_kinetic_energy_over_time)
        
        print(f"Evaluating the kinetic energy for a second time took {mean_time_kinetic_energy_over_time:.2e} seconds +/- {std_time_kinetic_energy_over_time:.2e} seconds (mean +/- std) over {nb_samples} samples. \n")
    
    # ===========================================================================
    # Save the simulation results
    # ===========================
    if bool_save_res:
        print("Saving the simulation results...")
        with open(results_path, "wb") as f:
            pickle.dump(simulation_dict, f)
        print(f"Simulation results saved at {results_path} \n")
    else:
        print("Simulation results not saved. \n")
        
    print("Simulation finished. \n")
    print("**************************************************************** \n \n")
        
    return simulation_dict