import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import imageio
def make_gif(filename,gifname):

    data=np.load(filename)
    T=data["T"]

    frames=[]

    for k in range(T.shape[0]):

        plt.figure(figsize=(4,4))
        plt.imshow(T[k],origin="lower",cmap="inferno")
        plt.colorbar()
        plt.title(f"Frame {k}")
        plt.tight_layout()

        plt.savefig("temp.png")
        plt.close()

        frames.append(imageio.imread("temp.png"))

    imageio.mimsave(gifname,frames,fps=10)
if __name__=="__main__":
    # generate_dataset()
    # make_gif("simulation_fast0.npz","simulation_fast0.gif")
    make_gif("simulation_fast1.npz","simulation_fast1.gif")