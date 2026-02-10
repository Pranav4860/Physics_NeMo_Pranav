import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------------
# Load synthetic data
# -----------------------------
data=np.load("rt_synthetic_sample.npz")
T=data["T"]

# -----------------------------
# Figure setup
# -----------------------------
fig,ax=plt.subplots()
im=ax.imshow(T[0],origin="lower",cmap="inferno")
ax.set_title("Rayleigh-Taylor Synthetic Simulation")
plt.colorbar(im,ax=ax)

# -----------------------------
# Animation update function
# -----------------------------
def update(frame):
    im.set_array(T[frame])
    ax.set_title(f"Time step {frame}")
    return [im]

# -----------------------------
# Create animation
# -----------------------------
ani=animation.FuncAnimation(
    fig,
    update,
    frames=T.shape[0],
    interval=50
)

# -----------------------------
# Save GIF
# -----------------------------
ani.save("rt_synthetic_simulation.gif",writer="pillow")

print("GIF saved as rt_synthetic_simulation.gif")