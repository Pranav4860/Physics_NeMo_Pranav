import numpy as np

# Grid params
Nx=64
Ny=64
Lx=1.0
Ly=1.0

x=np.linspace(0,Lx,Nx)
y=np.linspace(0,Ly,Ny)
X,Y=np.meshgrid(x,y,indexing="ij")

# Physical parameters

kappa=1e-3
Ra=1.0
dt=1e-3
nt=200


# Initial condition

def initial_temperature(X,Y):
    base=np.tanh((Y-0.5)*40)
    perturb=0.01*np.sin(2*np.pi*X)
    return base+perturb

T=initial_temperature(X,Y)


# Time integration

T_history=[]

for n in range(nt):
    T_history.append(T.copy())

    laplacian=(
        np.roll(T,1,axis=0)
        +np.roll(T,-1,axis=0)
        +np.roll(T,1,axis=1)
        +np.roll(T,-1,axis=1)
        -4*T
    )

    source=Ra*T*(1-T**2)

    T=T+dt*(kappa*laplacian+source)

T_history=np.array(T_history)


# Save data

np.savez(
    "rt_synthetic_sample.npz",
    T=T_history,
    x=x,
    y=y,
    dt=dt,
    Ra=Ra
)

print("Synthetic data generation complete")
