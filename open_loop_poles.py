import numpy as np
import matplotlib.pyplot as plt

# System Matrix A 
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [12.0073, -22.2863, 0, 0],
    [-14.4128, 94.3806, 0, 0]
])

# Compute Eigenvalues

eig_A = np.linalg.eigvals(A)

print("Eigenvalues of A:")
for val in eig_A:
    print(val)

# Plot Poles on s-plane

plt.figure(figsize=(6,5))

# Plot poles
plt.scatter(np.real(eig_A), np.imag(eig_A), 
            color='red', marker='x', s=120, label='Open-loop poles')

# Draw axes
plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

# Labels and title
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
plt.title('Open-Loop Poles on the s-plane')

# Annotate poles
for val in eig_A:
    plt.text(np.real(val) + 0.3, np.imag(val),
             f'{val:.2f}', fontsize=9)

# Grid and legend
plt.grid(True)
plt.legend()

# Save figure
plt.savefig("open_loop_poles.png", bbox_inches='tight')

# Show plot
plt.show()