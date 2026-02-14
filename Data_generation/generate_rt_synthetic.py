import numpy as np

def initial_conditions(Nx,Ny,Lx,Ly):

    x=np.linspace(0,Lx,Nx,endpoint=False)
    y=np.linspace(0,Ly,Ny)

    X,Y=np.meshgrid(x,y,indexing="ij")
    u=np.zeros((Nx,Ny))
    v=np.zeros((Nx,Ny))
    P=np.zeros((Nx,Ny))

    delta=0.02
    epsilon=0.01

    interface=0.5+epsilon*np.cos(2*np.pi*X)
    theta=np.tanh((Y-interface)/delta)
    return u,v,P,theta


