import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm

PRINT_DENSITY_MATRIX_ON_CHANGE = False

# Function to create a density matrix interpolating between a Bell state and a mixed state
def create_density_matrix(entanglement_degree):
    # Bell state (Phi+)
    bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho_bell = np.outer(bell_state, bell_state.conj())
    
    # Mixed state
    rho_mixed = 0.5 * np.diag([1, 0, 0, 1])

    # Interpolated density matrix
    rho = entanglement_degree * rho_bell + (1 - entanglement_degree) * rho_mixed
    return rho

# Function to apply rotation to a qubit
def rotate_qubit(rho, theta, qubit):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    if qubit == 1:
        R = np.kron(R, np.eye(2))
    else:
        R = np.kron(np.eye(2), R)

    if PRINT_DENSITY_MATRIX_ON_CHANGE:
        print(R @ rho @ R.T)
    return R @ rho @ R.T

# Function to calculate measurement probabilities
def measurement_probabilities(rho, x, y):
    # Measurement operators
    P_00 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    P_01 = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    P_10 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    P_11 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])

    probs = np.array([np.trace(P @ rho) for P in [P_00, P_01, P_10, P_11]])
    return probs

# Initial parameters
theta1, theta2 = 0.0, 0.0
entanglement_degree = 0.0
x, y = 0, 0  # Example inputs

# Create the initial density matrix
rho = create_density_matrix(entanglement_degree)
rho = rotate_qubit(rho, theta1, 1)
rho = rotate_qubit(rho, theta2, 2)
probs = measurement_probabilities(rho, x, y)

# Plotting
# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9)

# Bar plot for probabilities
bar_container = ax1.bar(range(4), probs, tick_label=["00", "01", "10", "11"])
ax1.set_ylim(0, 1)
ax1.set_ylabel('Probability')
ax1.set_title('Measurement Probabilities')

# Heat map for density matrix
rho = create_density_matrix(entanglement_degree)
rho = rotate_qubit(rho, theta1, 1)
rho = rotate_qubit(rho, theta2, 2)
cmap = cm.get_cmap('bwr')  # Blue-White-Red colormap
heatmap = ax2.imshow(rho.real, cmap=cmap, vmin=-1, vmax=1)
ax2.set_title('Density Matrix Before Measurement')
plt.colorbar(heatmap, ax=ax2)

text_box = ax2.text(0.5, 0.1, '', transform=ax2.transAxes, ha='center')

# Sliders
axcolor = 'lightgoldenrodyellow'
ax_theta1 = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_theta2 = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_entangle = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

slider_theta1 = Slider(ax_theta1, 'Theta1', -90, 90, valinit=theta1, valstep=2.5)
slider_theta2 = Slider(ax_theta2, 'Theta2', -90, 90, valinit=theta2, valstep=2.5)
slider_entangle = Slider(ax_entangle, 'Entanglement', 0, 1, valinit=entanglement_degree)

# Update function
def update(val):
    theta1 = np.pi * slider_theta1.val / 180
    theta2 = np.pi * slider_theta2.val / 180
    entanglement_degree = slider_entangle.val
    rho = create_density_matrix(entanglement_degree)
    rho = rotate_qubit(rho, theta1, 1)
    rho = rotate_qubit(rho, theta2, 2)
    probs = measurement_probabilities(rho, x, y)
    for rect, h in zip(bar_container.patches, probs):
        rect.set_height(h)
    heatmap.set_data(rho.real)
    fig.canvas.draw_idle()

# Connect the update function to the sliders
slider_theta1.on_changed(update)
slider_theta2.on_changed(update)
slider_entangle.on_changed(update)

plt.show()