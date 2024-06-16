import matplotlib.pyplot as plt
import numpy as np

# Redefine the correct 6x6 LMI matrix
matrix_6x6 = [
    ["-$\\mathcal{Y}$", "$(C\\mathcal{Y} + DU M)^T(R D_w + S^T)$", "-$\\lambda M^T S_w^T$", "$M^T X_+^T$", "$(C\\mathcal{Y} + DU M)^T$", "$M^T$"],
    ["$(C\\mathcal{Y} + DU M)^T(R D_w + S^T)^T$", "$Q + SD_w + D_w^T S^T + D_w^T R D_w$", "0", "$B_w^T$", "$B_w^T$", "0"],
    ["-$\\lambda S_w M$", "0", "$\\lambda Q_w$", "0", "0", "-$(\\lambda R_w)^{-1}$"],
    ["$X_+ M$", "$B_w$", "0", "-$\\mathcal{Y}$", "0", "0"],
    ["$C\\mathcal{Y} + DU M$", "$B_w$", "0", "0", "-$R^{-1}$", "0"],
    ["$M$", "0", "-$(\\lambda R_w)^{-1}$", "0", "0", "-$(\\lambda R_w)^{-1}$"]
]

# Define figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Hide the axes
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=matrix_6x6, cellLoc='center', loc='center', colWidths=[0.15]*6)

# Scale the font size for better readability
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)

# Set the table style
for key, cell in table.get_celld().items():
    cell.set_edgecolor('black')

# Save the table as a PNG image

plt.savefig('LMI_matrix.png')
plt.savefig('LMI_matrix.pdf')

plt.show()
