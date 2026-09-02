import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import jax.numpy as jnp
from data import folder_results
from space_convergence_study import refinements
from src.postprocessing.plot_config import configure_matplotlib

configure_matplotlib()  

folder_results_conv = folder_results + "space_convergence_Tend_1/"

refinement_overkill = refinements[-1]
r_mid_2_overkill = jnp.load(folder_results_conv + f"r_mid_2_refinement_{refinement_overkill}.npy")
refinements_wo_overkill = refinements[:-1]
error_t_final = []

for refinement in refinements_wo_overkill:
    print(f"Compute errors for refinement = {refinement}...")

    r_mid_2_h = jnp.load(folder_results_conv + f"r_mid_2_refinement_{refinement}.npy")

    error_i = jnp.linalg.norm(r_mid_2_h[:, -1] - r_mid_2_overkill[:, -1]) / jnp.linalg.norm(r_mid_2_overkill[:, -1])
    print(f"Error for refinement = {refinement}: {error_i}")

    error_t_final.append(error_i)

plt.figure()
plt.loglog(refinements_wo_overkill, error_t_final, 'o-')

# --- Empirical Convergence Rate Triangle (Middle Points, Reversed) ---
n_pts = len(refinements_wo_overkill)
if n_pts >= 2:
    # Find the two middle indices
    idx2 = n_pts // 2
    idx1 = idx2 - 1

    h1, h2 = refinements_wo_overkill[idx1], refinements_wo_overkill[idx2]
    e1, e2 = error_t_final[idx1], error_t_final[idx2]
    
    # Convergence rate formula: p = log(E2/E1) / log(h2/h1)
    rate = (jnp.log(e2) - jnp.log(e1)) / (jnp.log(h2) - jnp.log(h1))
    
    # Draw the reversed triangle legs (right angle at h1, e2)
    plt.plot([h1, h2], [e2, e2], 'k-', lw=1.2) # Horizontal base
    plt.plot([h1, h1], [e1, e2], 'k-', lw=1.2) # Vertical height
    
    # Add text labels for the triangle (using geometric means for log-scale centering)
    h_mid = jnp.sqrt(h1 * h2)
    e_mid = jnp.sqrt(e1 * e2)
    
    # Adjust text offsets: 1 is below the horizontal, p is to the right of the vertical
    plt.text(h_mid, e2 * 0.85, "1", ha='center', va='top')
    plt.text(h1 * 0.95, e_mid, f"{rate:.2f}", ha='right', va='center')
# ---------------------------------------------------------------------

# --- Clean X-Axis Tick Formatting ---
ax = plt.gca()

# 1. Explicitly place ticks only at your simulated time steps
ax.set_xticks(refinements_wo_overkill)

# 2. Format major ticks as standard decimal numbers (no scientific notation)
major_formatter = ticker.ScalarFormatter()
major_formatter.set_scientific(False)
ax.xaxis.set_major_formatter(major_formatter)

# 3. Disable minor tick labels completely to remove the overlap
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
# ------------------------------------

plt.xlabel('Refinement Factor')
plt.title('Relative Error')
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig(folder_results_conv + "space_convergence_plot.pdf", dpi=300)
plt.show()