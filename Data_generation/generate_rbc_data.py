import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import imageio


def create_grid(M,N,Lx,Ly):

    dx=Lx/M
    dy=Ly/N

    x=np.linspace(0,Lx,M,endpoint=False)   # periodic in x
    y=np.linspace(0,Ly,N)                  # physical walls in y
    return x,y,dx,dy

def create_fields(M,N,ng):   #ng: number of ghost cells layers
    
    shape=(M+2*ng,N+2*ng)

    u=np.zeros(shape)
    v=np.zeros(shape)
    P=np.zeros(shape)
    T=np.zeros(shape)

    return u,v,P,T

def initial_conditions(u,v,P,T,M,N,ng,x,y,Lx,Ly,n_modes):

    for i in range(M):
        for j in range(N):
            interface=0.02*np.cos(n_modes*x[i]/Lx*2.0*np.pi+0.2)+Ly/2.0
            T[ng+i,ng+j]=0.5*erf(-100.0*(y[j]-interface))+0.5

    u[ng:M+ng,ng:N+ng]=0.0
    v[ng:M+ng,ng:N+ng]=0.0
    P[ng:M+ng,ng:N+ng]=0.0

    return u,v,P,T


def boundary_uv(u,v,M,N,ng):

    # periodic in x
    u[0,:]=u[M+ng-1,:]
    u[M+ng,:]=u[ng,:]

    v[0,:]=v[M+ng-1,:]
    v[M+ng,:]=v[ng,:]

    # bottom wall
    v[:,ng-1]=0.0
    u[:,ng-1]=-u[:,ng]

    # top wall
    v[:,N+ng-1]=0.0
    u[:,N+ng]=-u[:,N+ng-1]

    return u,v

def boundary_T(T,M,N,ng):

    # periodic in x
    T[0,:]=T[M+ng-1,:]
    T[M+ng,:]=T[ng,:]

    # bottom T=1
    T[:,ng-1]=2.0-T[:,ng]

    # top T=0
    T[:,N+ng]=-T[:,N+ng-1]

    return T

def boundary_P(P,M,N,ng):
    P[0,:]=P[M+ng-1,:]
    P[M+ng,:]=P[ng,:]
    return P

def divergence(u,v,M,N,ng,dx,dy):

    i0,i1=ng,ng+M
    j0,j1=ng,ng+N

    D=(u[i0:i1,j0:j1]-u[i0-1:i1-1,j0:j1])/dx + \
      (v[i0:i1,j0:j1]-v[i0:i1,j0-1:j1-1])/dy

    return D

def tridiag(a, b, c, r):
    """
    Thomas algorithm.
    a,b,c,r are 1D arrays of length n.
    a[0] should be 0, c[-1] should be 0.
    """
    n = r.size
    gam = np.zeros_like(r, dtype=np.complex128)
    u = np.zeros_like(r, dtype=np.complex128)

    if np.abs(b[0]) < np.finfo(float).eps:
        raise RuntimeError("tridiag: error1 (small first diagonal)")

    bet = b[0]
    u[0] = r[0] / bet

    for j in range(1, n):
        gam[j] = c[j - 1] / bet
        bet = b[j] - a[j] * gam[j]
        if np.abs(bet) < np.finfo(float).eps:
            raise RuntimeError("tridiag: error2 (zero pivot)")
        u[j] = (r[j] - a[j] * u[j - 1]) / bet

    for j in range(n - 2, -1, -1):
        u[j] = u[j] - gam[j + 1] * u[j + 1]

    return u

def main_poisson_solve(D,M,N,dx,dy):

    Dhat=np.fft.fft(D,n=M,axis=0)
    Phat=np.zeros_like(Dhat,dtype=np.complex128)

    for p in range(1,M):
        k=p
        r=Dhat[p,:]

        a=(1.0/dy**2)*np.ones(N,dtype=np.complex128)
        c=(1.0/dy**2)*np.ones(N,dtype=np.complex128)
        a[0]=0.0
        c[-1]=0.0

        Cb=(-2.0/dy**2)+(1.0/dx**2)*(-2.0+(2.0*np.cos(2.0*np.pi*k/M)))

        b=np.empty(N,dtype=np.complex128)
        b[1:-1]=Cb
        b[0]=Cb+(1.0/dy**2)
        b[-1]=Cb+(1.0/dy**2)

        Phat[p,:]=tridiag(a,b,c,r)

    Phat[0,:]=0.0

    P=np.fft.ifft(Phat,n=M,axis=0).real
    return P

def vmean_zero(u,v,M,N,ng):

    i0,i1=ng,ng+M

    vxmean=v[i0:i1,:].mean(axis=0)
    v=v-vxmean[None,:]

    u,v=boundary_uv(u,v,M,N,ng)

    return u,v

def div_free_vel(u,v,P,M,N,ng,dx,dy):

    i0,i1=ng,ng+M
    j0,j1=ng,ng+N

    u,v=vmean_zero(u,v,M,N,ng)

    D=divergence(u,v,M,N,ng,dx,dy)

    P[i0:i1,j0:j1]=main_poisson_solve(D,M,N,dx,dy)

    P=boundary_P(P,M,N,ng)

    u[i0:i1,j0:j1]=u[i0:i1,j0:j1] - \
        (P[i0+1:i1+1,j0:j1]-P[i0:i1,j0:j1])/dx

    v[i0:i1,j0:j1-1]=v[i0:i1,j0:j1-1] - \
        (P[i0:i1,j0+1:j1]-P[i0:i1,j0:j1-1])/dy

    u,v=boundary_uv(u,v,M,N,ng)

    return u,v,P


# def fluxes(u,v,T,M,N,ng):

#     fu=np.zeros_like(u)
#     gu=np.zeros_like(u)
#     fv=np.zeros_like(u)
#     gv=np.zeros_like(u)
#     fT=np.zeros_like(u)
#     gT=np.zeros_like(u)

#     for i in range(0,M+2*ng):
#         for j in range(0,N+2*ng):

#             if (i-1)>=0:
#                 fu[i,j]=((u[i-1,j]+u[i,j])**2)/4.0
#                 fT[i,j]=(T[i-1,j]+T[i,j])/2.0*(u[i-1,j])

#             if (j-1)>=0 and (i+1)<=M+2*ng-1:
#                 gu[i,j]=(u[i,j]+u[i,j-1])*(v[i,j-1]+v[i+1,j-1])/4.0

#             if (i-1)>=0 and (j+1)<=N+2*ng-1:
#                 fv[i,j]=(u[i-1,j]+u[i-1,j+1])*(v[i-1,j]+v[i,j])/4.0

#             if (j-1)>=0:
#                 gv[i,j]=((v[i,j-1]+v[i,j])**2)/4.0
#                 gT[i,j]=(T[i,j-1]+T[i,j])/2.0*(v[i,j-1])

#     return fu,gu,fv,gv,fT,gT
def fluxes(u,v,T,M,N,ng):

    fu=np.zeros_like(u)
    gu=np.zeros_like(u)
    fv=np.zeros_like(u)
    gv=np.zeros_like(u)
    fT=np.zeros_like(u)
    gT=np.zeros_like(u)

    # fu and fT (i-1,i)
    fu[1:,:]=((u[:-1,:]+u[1:,:])**2)/4.0
    fT[1:,:]=(T[:-1,:]+T[1:,:])/2.0*u[:-1,:]

    # gu
    gu[:-1,1:]=(u[:-1,1:]+u[:-1,:-1])*(v[:-1,:-1]+v[1:,:-1])/4.0

    # fv
    fv[1:,:-1]=(u[:-1,:-1]+u[:-1,1:])*(v[:-1,:-1]+v[1:,:-1])/4.0

    # gv and gT
    gv[:,1:]=((v[:,:-1]+v[:,1:])**2)/4.0
    gT[:,1:]=(T[:,:-1]+T[:,1:])/2.0*v[:,:-1]

    return fu,gu,fv,gv,fT,gT


def rhs_conv(u,v,T,M,N,ng,dx,dy):

    i0,i1=ng,ng+M
    j0,j1=ng,ng+N

    ru=np.zeros_like(u)
    rv=np.zeros_like(u)
    rt=np.zeros_like(u)

    fu,gu,fv,gv,fT,gT=fluxes(u,v,T,M,N,ng)

    dfudx=(fu[i0+1:i1+1,j0:j1]-fu[i0:i1,j0:j1])/dx
    dgudy=(gu[i0:i1,j0+1:j1+1]-gu[i0:i1,j0:j1])/dy

    dfvdx=(fv[i0+1:i1+1,j0:j1]-fv[i0:i1,j0:j1])/dx
    dgvdy=(gv[i0:i1,j0+1:j1+1]-gv[i0:i1,j0:j1])/dy

    dfTdx=(fT[i0+1:i1+1,j0:j1]-fT[i0:i1,j0:j1])/dx
    dgTdy=(gT[i0:i1,j0+1:j1+1]-gT[i0:i1,j0:j1])/dy

    ru[i0:i1,j0:j1]=-dfudx-dgudy
    rv[i0:i1,j0:j1]=-dfvdx-dgvdy
    rt[i0:i1,j0:j1]=-dfTdx-dgTdy

    return ru,rv,rt

# def rhs_vis(u,v,T,M,N,ng,dx,dy,Pr):

#     i0,i1=ng,ng+M
#     j0,j1=ng,ng+N

#     ru=np.zeros_like(u)
#     rv=np.zeros_like(u)
#     rt=np.zeros_like(u)

#     for i in range(i0,i1):
#         for j in range(j0,j1):

#             ru[i,j]=((u[i-1,j]-2*u[i,j]+u[i+1,j])/dx**2 +
#                      (u[i,j-1]-2*u[i,j]+u[i,j+1])/dy**2)

#             rv[i,j]=((v[i-1,j]-2*v[i,j]+v[i+1,j])/dx**2 +
#                      (v[i,j-1]-2*v[i,j]+v[i,j+1])/dy**2)

#             rt[i,j]=((T[i-1,j]-2*T[i,j]+T[i+1,j])/dx**2 +
#                      (T[i,j-1]-2*T[i,j]+T[i,j+1])/dy**2)

#     ru*=Pr
#     rv*=Pr

#     return ru,rv,rt
def rhs_vis(u,v,T,M,N,ng,dx,dy,Pr):

    i0,i1=ng,ng+M
    j0,j1=ng,ng+N

    ru=np.zeros_like(u)
    rv=np.zeros_like(u)
    rt=np.zeros_like(u)

    # second derivatives (central difference)

    ru[i0:i1,j0:j1]=(
        (u[i0-1:i1-1,j0:j1]-2*u[i0:i1,j0:j1]+u[i0+1:i1+1,j0:j1])/dx**2 +
        (u[i0:i1,j0-1:j1-1]-2*u[i0:i1,j0:j1]+u[i0:i1,j0+1:j1+1])/dy**2
    )

    rv[i0:i1,j0:j1]=(
        (v[i0-1:i1-1,j0:j1]-2*v[i0:i1,j0:j1]+v[i0+1:i1+1,j0:j1])/dx**2 +
        (v[i0:i1,j0-1:j1-1]-2*v[i0:i1,j0:j1]+v[i0:i1,j0+1:j1+1])/dy**2
    )

    rt[i0:i1,j0:j1]=(
        (T[i0-1:i1-1,j0:j1]-2*T[i0:i1,j0:j1]+T[i0+1:i1+1,j0:j1])/dx**2 +
        (T[i0:i1,j0-1:j1-1]-2*T[i0:i1,j0:j1]+T[i0:i1,j0+1:j1+1])/dy**2
    )

    ru*=Pr
    rv*=Pr

    return ru,rv,rt

def rhs_buo(T,M,N,ng,Ra,Pr):

    i0,i1=ng,ng+M
    j0,j1=ng,ng+N

    ru=np.zeros_like(T)
    rv=np.zeros_like(T)
    rt=np.zeros_like(T)

    rv[i0:i1,j0:j1]=Ra*Pr*(T[i0:i1,j0:j1] +
                           T[i0:i1,j0+1:j1+1])/2.0

    return ru,rv,rt


def rhs(u,v,T,M,N,ng,dx,dy,Ra,Pr):

    ru_c,rv_c,rt_c=rhs_conv(u,v,T,M,N,ng,dx,dy)
    ru_v,rv_v,rt_v=rhs_vis(u,v,T,M,N,ng,dx,dy,Pr)
    ru_b,rv_b,rt_b=rhs_buo(T,M,N,ng,Ra,Pr)

    ru=ru_c+ru_v+ru_b
    rv=rv_c+rv_v+rv_b
    rt=rt_c+rt_v+rt_b

    return ru,rv,rt


def update(dt,a,u,r0,r1,r2):
    return u+dt*(a[0]*r0+a[1]*r1+a[2]*r2)


def RungeKutta_step(u,v,P,T,dt,M,N,ng,dx,dy,Ra,Pr):

    ru0,rv0,rt0=rhs(u,v,T,M,N,ng,dx,dy,Ra,Pr)

    u1=update(dt,[0.5,0,0],u,ru0,0,0)
    v1=update(dt,[0.5,0,0],v,rv0,0,0)
    T1=update(dt,[0.5,0,0],T,rt0,0,0)

    T1=boundary_T(T1,M,N,ng)
    u1,v1,P1=div_free_vel(u1,v1,P.copy(),M,N,ng,dx,dy)

    ru1,rv1,rt1=rhs(u1,v1,T1,M,N,ng,dx,dy,Ra,Pr)

    u2=update(dt,[-1,2,0],u,ru0,ru1,0)
    v2=update(dt,[-1,2,0],v,rv0,rv1,0)
    T2=update(dt,[-1,2,0],T,rt0,rt1,0)

    T2=boundary_T(T2,M,N,ng)
    u2,v2,P2=div_free_vel(u2,v2,P.copy(),M,N,ng,dx,dy)

    ru2,rv2,rt2=rhs(u2,v2,T2,M,N,ng,dx,dy,Ra,Pr)

    u_new=update(dt,[1/6,2/3,1/6],u,ru0,ru1,ru2)
    v_new=update(dt,[1/6,2/3,1/6],v,rv0,rv1,rv2)
    T_new=update(dt,[1/6,2/3,1/6],T,rt0,rt1,rt2)

    T_new=boundary_T(T_new,M,N,ng)
    u_new,v_new,P_new=div_free_vel(u_new,v_new,P.copy(),M,N,ng,dx,dy)

    return u_new,v_new,P_new,T_new

def simulate(Ra,Pr,M,N,ng,Lx,Ly,dt,n_steps,save_every):

    x,y,dx,dy=create_grid(M,N,Lx,Ly)

    u,v,P,T=create_fields(M,N,ng)

    u,v,P,T=initial_conditions(u,v,P,T,M,N,ng,x,y,Lx,Ly,n_modes=2)

    T=boundary_T(T,M,N,ng)
    u,v=boundary_uv(u,v,M,N,ng)

    snapshots_T=[]
    snapshots_u=[]
    snapshots_v=[]

    for step in range(n_steps):

        u,v,P,T=RungeKutta_step(u,v,P,T,dt,M,N,ng,dx,dy,Ra,Pr)

        if step%save_every==0:
            snapshots_T.append(T[ng:M+ng,ng:N+ng].copy())
            snapshots_u.append(u[ng:M+ng,ng:N+ng].copy())
            snapshots_v.append(v[ng:M+ng,ng:N+ng].copy())

    return np.array(snapshots_T),np.array(snapshots_u),np.array(snapshots_v)


def generate_dataset():

    M=128
    N=128
    
    ng=1

    Lx=1.0
    Ly=1.0

    Pr=1.0
    dt=1e-6
    n_steps=23000
    save_every=200   # saves 100 frames approx

    Ra_values=[1e4,3e4]   # only 2 Ra

    for i,Ra in enumerate(Ra_values):

        print("Running Ra =",Ra)

        T_data,u_data,v_data=simulate(Ra,Pr,M,N,ng,Lx,Ly,dt,n_steps,save_every)

        np.savez(f"simulation_fast{i}.npz",
                 T=T_data,
                 u=u_data,
                 v=v_data,
                 Ra=Ra,
                 Pr=Pr)

        print("Saved simulation_",i)

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
    generate_dataset()
    make_gif("simulation_0.npz","simulation_0.gif")
    make_gif("simulation_1.npz","simulation_1.gif")

