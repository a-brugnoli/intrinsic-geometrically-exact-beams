import jax.numpy as jnp
from run_four_bar_gyrator import run_four_bar_gyrator
from data import n_elem_2_base

dt = 0.02

refinements = [1, 2, 3, 4, 16]


if __name__ == "__main__":
    import os
    from data import folder_results
    T_end = int(1)
    folder_results_conv = folder_results + f"space_convergence_Tend_{T_end}/"
    if not os.path.exists(folder_results_conv):
        os.makedirs(folder_results_conv)
    for rfactor in refinements:
        print(f"Running simulation for refinement factor = {rfactor}...")

        positions_sol_refinement = run_four_bar_gyrator(dt, T_end, folder_results_conv, \
                                                refinement_factor=rfactor,
                                                compute_positions=True, save_data=False)[-1]

        position_2_sol_refinement = positions_sol_refinement[1]

        n_elem_2 = n_elem_2_base * rfactor
        node_mid_2 = int(n_elem_2/2)

        r_mid_2_refinement = position_2_sol_refinement[:, node_mid_2, :]

        jnp.save(f"{folder_results_conv}r_mid_2_refinement_{rfactor}.npy", r_mid_2_refinement)

    print("Space convergence study data saved successfully.")
