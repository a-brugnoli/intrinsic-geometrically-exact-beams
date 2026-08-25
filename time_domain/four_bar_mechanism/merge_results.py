import numpy as np
from pathlib import Path
import os
path_file = os.path.dirname(os.path.abspath(__file__))

folder_results = path_file + "/results/gyrator_stiff/"

# Load data
iters = np.load(folder_results + "iterations.npy")

# Read original TikZ file
tikz = Path(folder_results + "iterations_kinon.tikz").read_text()

# Build PGFPlots table
table_lines = [f"                {i+1}\t{v}\\\\" for i, v in enumerate(iters)]

plot_block = (
    "\n"
    "\\addplot [color=magenta, line width = 1.2pt]\n"
    "        table[row sep=crcr]{%\n"
    + "\n".join(table_lines)
    + "\n        };\n"
    "\\addlegendentry{PH intrinsic}\n"
)

# Insert before the end of the axis environment
tikz_new = tikz.replace("\\end{axis}", plot_block + "\\end{axis}")

# Save
Path(folder_results + "newton_iterations_EM.tikz").write_text(tikz_new)