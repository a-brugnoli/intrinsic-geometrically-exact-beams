import jax.numpy as jnp
from run_four_bar_gyrator import run_four_bar_gyrator
from data import n_elem_2_base
node_mid_2 = int(n_elem_2_base/2)

time_steps = [0.1, 0.05, 0.025, 0.0125, 0.005]

if __name__ == "__main__":
    import os
    from data import folder_results
    T_end = int(1)
    folder_results_conv = folder_results + f"time_convergence_Tend_{T_end}/"
    if not os.path.exists(folder_results_conv):
        os.makedirs(folder_results_conv)
    for dt in time_steps:
        print(f"Running simulation for dt = {dt}...")

        positions_sol_dt = run_four_bar_gyrator(dt, T_end, folder_results_conv, \
                                                compute_positions=True, save_data=False)[-1]

        position_2_sol_dt = positions_sol_dt[1]
        r_mid_2_dt = position_2_sol_dt[:, node_mid_2, :]

        dt_str = f"{dt:g}"
        jnp.save(f"{folder_results_conv}r_mid_2_dt_{dt_str}.npy", r_mid_2_dt)

    print("Convergence study data saved successfully.")
