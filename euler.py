import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Simulation Parameters
# ------------------------
N = 256           # Number of grid points per axis
L = 100.0         # Spatial extent per axis
dx = L / N        # Spatial step
x = np.linspace(-L/2, L/2, N)

# Create 2D grids for the two particles (x1 and x2)
X1, X2 = np.meshgrid(x, x, indexing='ij')

# Time evolution parameters
dt = 0.005       # Time step (may need to be very small for stability)
num_steps = 5000  # Number of time steps

# Physical parameters (using ℏ = 1, m = 1)
m = 1.0
sigma = 3.0

# ------------------------
# One-Particle Wavefunctions
# ------------------------
p0_1 = - 3.0
p0_2 = 3.0

# Define one-particle states on the 1D grid
phi1 = (np.exp(1j * p0_1 * x) + np.exp(-1j * p0_1 * x)) * np.exp(-x**2 / (4 * sigma**2))
phi2 = (np.exp(1j * p0_2 * x) + np.exp(-1j * p0_2 * x)) * np.exp(-x**2 / (4 * sigma**2))


phi1 = np.exp(1j * p0_1 * x) * np.exp(-(x+10)**2 / (4 * sigma**2)) + np.exp(-1j * p0_1 * x) * np.exp(-(x-20)**2 / (4 * sigma**2))
phi2 = np.exp(1j * p0_1 * x) * np.exp(-(x+20)**2 / (4 * sigma**2)) + np.exp(-1j * p0_1 * x) * np.exp(-(x-30)**2 / (4 * sigma**2))

# Normalize the one-particle states
norm1 = np.sqrt(np.sum(np.abs(phi1)**2) * dx)
norm2 = np.sqrt(np.sum(np.abs(phi2)**2) * dx)
phi1 /= norm1
phi2 /= norm2

# ------------------------
# Construct the Two-Particle State
# ------------------------
# For example, an entangled state:
# psi = (np.outer(phi1, phi2) + np.outer(phi2, phi1)) / np.sqrt(2) # the two particles are interchangable
psi = np.outer(phi1, phi2)
# psi is a 2D array representing psi(x1,x2)

# ------------------------
# Finite Difference Laplacian
# ------------------------
def laplacian(psi, dx):
    """
    Compute the Laplacian of a 2D array psi using finite differences.
    Uses periodic boundary conditions via np.roll.
    """
    lap = (np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
           np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi) / dx**2
    return lap

# ------------------------
# Time Evolution (Explicit Euler)
# ------------------------
def time_step(psi, dt, dx):
    """
    Evolve psi by one time step using the explicit Euler method.
    Schrödinger eq: i dψ/dt = -(1/2) ∇²ψ  ->  dψ/dt = -i/2 ∇²ψ
    """
    return psi - 1j * dt/2 * laplacian(psi, dx)


joint_prob = np.abs(psi)**2

# ------------------------
# Initial Plot
# ------------------------
do_initial_plot = False
if do_initial_plot:
    plt.figure(figsize=(6,5))
    plt.imshow(joint_prob, extent=[x[0], x[-1], x[0], x[-1]], origin='lower', aspect='auto')
    plt.title("Initial Joint Probability |ψ(x₁,x₂)|²")
    plt.xlabel("x₂")
    plt.ylabel("x₁")
    plt.colorbar()
    plt.show()

# ------------------------
# Time Evolution Loop
# ------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(joint_prob, extent=[x[0], x[-1], x[0], x[-1]], origin='lower', aspect='auto')
ax.set_title("Joint Probability |ψ(x₁,x₂)|²")
ax.set_xlabel("x₂")
ax.set_ylabel("x₁")
plt.colorbar(im, ax=ax)

for step in range(num_steps):
    psi = time_step(psi, dt, dx)
    if step % 20 == 0:
        joint_prob = np.abs(psi)**2
        im.set_data(joint_prob)
        im.set_clim(joint_prob.min(), joint_prob.max())
        plt.pause(0.01)

plt.ioff()
plt.show()
