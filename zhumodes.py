# optical modesolver from Zhu and Brown, Full-vectorial finite-difference analysis of microstructured optical fibers, 2002
# Zhu2002 is also used by philsol https://github.com/philmain28/philsol and Ansys, see https://optics.ansys.com/hc/en-us/articles/360034917233-MODE-Finite-Difference-Eigenmode-FDE-solver-introduction

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams['keymap.quit'] = ['ctrl+w','cmd+w','q','escape']

def matrixform(k0,n,dx,dy,bcx=None,bcy=None,bcvisualize=False):
    import scipy.sparse as sps
    # boundary conditions are periodic by default
    # boundary conditions are specified for Ey field, which are opposite for Ex for dirichlet and neumann
    # p=periodic, a,d=antisymmetric/dirichlet, s,n=symmetric/neumann
    bcx,bcy = bcx.lower() if bcx is not None else 'p',bcy.lower() if bcy is not None else 'p'
    ni,nj,_ = np.shape(n)
    ux = ( -np.eye(nj,k=0) + np.eye(nj,k=1) )
    if bcy in ['p','periodic']:
        ux[nj-1,0] = 1
    elif bcy in ['a','d','dirichlet']: # for Ey
        ux[0,0] = 0 # Ex S, Ey A
    elif bcy in ['s','n','neumann']: # for Ey
        ux[-1,-1] = 0 # Ex A, Ey S
    else:
        raise ValueError(f'unknown boundary condition bcy {bcy}')
    uxblocks = [ux for i in range(ni)]
    Ux = sps.block_diag(uxblocks,format='csr') # print('Ux.getnnz()',Ux.getnnz()) # number of stored values, including explicit zeros
    if bcx in ['p','periodic']:
        Uy = -sps.eye(nj*ni,k=0) + sps.eye(nj*ni,k=nj) + sps.eye(nj*ni,k=-nj*(ni-1))
    elif bcx in ['a','d','dirichlet']: # for Ey
        Uy = -sps.diags(np.where(np.arange(nj*ni)<nj*(ni-1),+1,0),format='csr') + sps.eye(nj*ni,k=nj,format='csr')
    elif bcx in ['s','n','neumann']: # for Ey
        Uy = -sps.diags(np.where(np.arange(nj*ni)<nj,0,+1),format='csr') + sps.eye(nj*ni,k=nj,format='csr')
    else:
        raise ValueError(f'unknown boundary condition bcx {bcx}')
    Ux,Uy = Ux/dx,Uy/dy
    Vx,Vy = -Ux.transpose(),-Uy.transpose()
    I =  sps.eye(nj*ni)
    # if bcvisualize:
    #     from wavedata import Wave2D
    #     Ux,Uy = Ux*dx,Uy*dy
    #     xs,ys,dd = dx*np.arange(nj),dy*np.arange(ni),dict(colormesh=1,aspect=1)
    #     nn = (n[:,:,0].flatten()).reshape(*n.shape[0:2],)
    #     Wave2D(nn,xs,ys).plot(legendtext='index',**dd)
    #     Wave2D(((Ux+Uy) * n[:,:,0].flatten()).reshape(*n.shape[0:2],) ,xs,ys).plot(legendtext='Ux+Uy',**dd)
    #     Wave2D((n[:,:,0].flatten() * (Vx+Vy)).reshape(*n.shape[0:2],) ,xs,ys).plot(legendtext='Vx+Vy',**dd)
    #     Ux,Uy = Ux/dx,Uy/dy
    εx = (n[:,:,0]**2).flatten()
    εy = (n[:,:,1]**2).flatten()
    εzinv = (n[:,:,2]**(-2)).flatten()
    εx =  sps.spdiags(εx,0,nj*ni,nj*ni,format='csr')
    εy =  sps.spdiags(εy,0,nj*ni,nj*ni,format='csr')
    εzinv = sps.spdiags(εzinv,0,nj*ni,nj*ni,format='csr')
    Pxx = (-Ux * εzinv * Vy * Vx * Uy / k0**2 + (k0**2 * I + Ux * εzinv * Vx) * (εx + Vy * Uy / k0**2) )
    Pyy = (-Uy * εzinv * Vx * Vy * Ux / k0**2 + (k0**2 * I + Uy * εzinv * Vy) * (εy + Vx * Ux / k0**2) )
    Pxy = ( Ux * εzinv * Vy * (εy + Vx * Ux / k0**2) - (k0**2 * I + Ux * εzinv * Vx) * Vy * Ux / k0**2 )
    Pyx = ( Uy * εzinv * Vx * (εx + Vy * Uy / k0**2) - (k0**2 * I + Uy * εzinv * Vy) * Vx * Uy / k0**2 )
    P = sps.vstack([ sps.hstack([Pxx,Pxy]), sps.hstack([Pyx,Pyy]) ]) 
    return P, εx, εy, εzinv, Ux, Uy, Vx, Vy
def methodindex(n,method=None):
    if method in ('exact','supress',None): return n.copy()
    assert method in ['teisotropic','tmisotropic'], f"unknown method {method}"
    nn = n.copy()
    nn[:,:,:] = n[:,:,0:1] if 'teisotropic'==method else n[:,:,1:2]
    return nn
def zhumodesolve(λ,n,x,y,nmodes,nguess=None,method=None,boundary=None,check=True): # boundary: None,'periodic','dirichlet','neumann','p','d','n'
    # ZhuBrown2002 Eqns. 7 and 8
    #   ik₀ Hx = -iβ Ey + Uy Ez
    #   ik₀ Hy =  iβ Ex - Ux Ez
    #   ik₀ Hz = -Uy Ex + Ux Ey
    #   -ik₀εx Ex = -iβ Hy + Vy Hz
    #   -ik₀εy Ey =  iβ Hx - Vx Hz
    #   -ik₀εz Ez = -Vy Hx + Vx Hy
    from scipy.sparse.linalg import eigs
    k = 2*np.pi/λ
    nguess = nguess if nguess is not None else n.max()
    # print('nguess',nguess)
    bcx,bcy = boundary if not isinstance(boundary,str) and hasattr(boundary,'__len__') else (boundary,boundary)
    P,εx,εy,εzinv,Ux,Uy,Vx,Vy = matrixform(k0=k,n=methodindex(n,method),dx=x[1]-x[0],dy=y[1]-y[0],bcx=bcx,bcy=bcy)
    betaguess = 2*np.pi*nguess/λ
    betasqr,E = eigs(P,k=nmodes,sigma=betaguess**2,maxiter=None)
    β = betasqr**0.5
    Ex,Ey = np.split(E, 2)
    Hz = (-Uy*Ex + Ux*Ey)/(1j*k)
    Hy = (-1j*k*εx*Ex - Vy*Hz)/(-1j*β)
    Hx = (-1j*k*εy*Ey + Vx*Hz)/(1j*β)
    Ez = εzinv*(-Vy*Hx + Vx*Hy)/(-1j*k)
    if check:
        assert np.allclose(1j*k*Hx, -1j*β*Ey + Uy*Ez)
        assert np.allclose(1j*k*Hy,  1j*β*Ex - Ux*Ez)
        assert np.allclose(1j*k*Hz, -Uy*Ex + Ux*Ey)
        assert np.allclose(-1j*k*εx*Ex, -1j*β*Hy + Vy*Hz)
        assert np.allclose(-1j*k*εy*Ey,  1j*β*Hx - Vx*Hz)
        assert np.allclose(-1j*k*Ez, εzinv*(-Vy*Hx + Vx*Hy))
    def convert(E): # convert 1D form to 2D
        return [np.reshape(e, (len(x), len(y))) for e in np.transpose(E)]
    neffs = [0.5*k*λ/np.pi for k in β]
    Exs,Eys,Ezs,Hxs,Hys,Hzs = [convert(E) for E in [Ex,Ey,Ez,Hx,Hy,Hz]]
    # print('n.shape',n.shape,'Ex.shape',Exs[0].shape)
    return neffs,Exs,Eys,Ezs,Hxs,Hys,Hzs
def cupytest():
    ## cupy gpu mode won't work until cupy.linalg.eig becomes available
    import sys
    print('running python version',sys.version)
    import cupy as cp
    # Stable implementation of log(1 + exp(x)) example
    def softplus(x):
        xp = cp.get_array_module(x)  # 'xp' is a standard usage in the community
        print("Using:", xp.__name__)
        return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))
    print(softplus(cp.array([1])))
    x = cp.arange(6).reshape(2, 3).astype('f')
    print(x)
    print(x.sum(axis=1))
    import cupyx.scipy.sparse as cpsparse
    A = cpsparse.random(1000, 1000, density=0.01, format='csr')
    try:
        from cupy.linalg import eig
        eigenvalues, eigenvectors = eig(A, k=6, which='LM')
        print(eigenvalues)
        print(eigenvectors.shape)
    except ImportError as e:
        print(e)
        print('cupy.linalg.eig not available')
def plotindex(n,x,y,axis=1,clim=None,aspect=1):
    clim = clim if clim is not None else (n.max()-0.03,n.max())
    plt.pcolor(x,y,np.transpose(n[:,:,axis]), cmap=cm.Blues_r)
    plt.clim(clim)
    plt.colorbar()
    plt.gca().set_aspect(aspect)
    plt.show()
def modeplot(ee,n,x,y,axis=1,ncontours=49,aspect=1):
    plt.figure()
    plt.pcolor(x,y,np.transpose(n[:,:,axis]), cmap=cm.Blues_r)
    plt.clim((n.max()-0.03,n.max()))
    plt.colorbar()
    plt.gca().set_aspect(aspect)
    e = np.transpose(ee.real)
    levels = np.linspace(np.min(e), np.max(e), ncontours+2)
    c_plt = plt.contour(x, y, e, cmap=cm.inferno, levels=levels)
    plt.show()
def isotropicindex(num_x=81,num_y=101,w=8.,h=10.,h0=1.):
    n = np.ones((num_x, num_y, 3))*1.8 # + 1j*1e-3
    # n[-71:,20:60,:] = n[-71:,20:60,:] + 0.02; n[-11:,:,:] = 1
    n[20:60,-71:,:] = n[20:60,-71:,:] + 0.02
    n[:,-11:,:] = 1
    x = np.linspace(-w/2.,w/2.,num_x)
    y = np.linspace(h0-h,h0,num_y)
    # print('n.shape',n.shape) # (81, 101, 3)
    return n,x,y
def isotropicexample(λ=0.3,plot=False):
    n,x,y = isotropicindex()
    # if plot: plotindex(n,x,y)
    # neffs,eexs,eeys = zhumodesolve(λ,n,x,y,nmodes=10,nguess=1.82)
    neffs,eexs,eeys,*_ = zhumodesolve(λ,n,x,y,nmodes=10,nguess=None)
    for i in range(len(neffs)):
        print(f"effective index {neffs[i]} polarization: {'x' if abs(eexs[i]).max()>abs(eeys[i]).max() else 'y'}")
        ee = eexs[i] if abs(eexs[i]).max()>abs(eeys[i]).max() else eeys[i]
        if plot: modeplot(ee,n,x,y)
def filtermodes(neffs,eexs,eeys,polarization='tm'):
    # return neffs,ees only for ees with correct polarization
    polarization = polarization.lower()
    assert polarization in 'h v te tm'.split()
    polarization = {'h':'te','v':'tm'}[polarization] if polarization in 'hv' else polarization
    def volume(E):
        return (np.abs(E)**2).mean()
    def istm(Ex,Ey):
        return volume(Ex)<volume(Ey)
    def iscorrectpolarization(Ex,Ey):
        return istm(Ex,Ey) if 'tm'==polarization else not istm(Ex,Ey)
    def correctE(Ex,Ey):
        return Ey if 'tm'==polarization else Ex
    # print('polarization',polarization)
    # print('iscorrect',[iscorrectpolarization(Ex,Ey) for neff,Ex,Ey in zip(neffs,eexs,eeys)])
    return list(zip(*[(neff,correctE(Ex,Ey)) for neff,Ex,Ey in zip(neffs,eexs,eeys) if iscorrectpolarization(Ex,Ey)]))
def anisotropicindex(dn,numx=81,numy=101,w=8.,h=10.,h0=1.):
    n = np.zeros((numx, numy, 3))
    n[:,:,0] += 1.7 # axis 0 is horizontal  (ny = ~1.7)
    n[:,:,1] += 1.8 # axis 1 is vertical    (nz = ~1.8)
    n[:,:,2] += 1.6 # axis 2 is propagation (nx = ~1.6)
    n[20:60,-71:,:] += dn
    n[:,-11:,:] = 1 # air above
    # n[-int(0.7*numy):,int(0.25*numx):int(0.75*numx),:] += dn
    # n[-int(0.1*numy):,:,:] = 1
    # n[:,int(0.25*numx):int(0.75*numx),:] += dn    
    # n[:,int(0.25*numx):int(0.5*numx),:] += dn
    # n[int(0.25*numy):int(0.5*numy),:,:] += dn
    # n[int(0.25*numy):int(0.75*numy),-int(0.7*numx):,:] += dn
    # n[:,-int(0.1*numx):,:] = 1
    # n[int(0.25*numy):int(0.65*numy),-int(0.7*numx):-int(0.3*numx),:] += dn
    # n[int(0.25*numy):int(0.75*numy),:,:] += dn
    # n[int(0.05*numy):int(0.75*numy),-int(0.7*numx):,:] += dn
    x = np.linspace(-w/2.,w/2.,numx)
    y = np.linspace(h0-h,h0,numy)
    # print('n.shape',n.shape,'n.shape==(81,101,3)',n.shape==(81,101,3))
    return n,x,y
def anisotropicexample(λ=1.5,dn=0.01,nmodes=10,nguess=None,polarization=None,boundary=None,plot=False,method=None):
    n,x,y = anisotropicindex(dn)
    # if plot: plotindex(n,x,y)
    # neffs,eexs,eeys = zhumodesolve(λ,n,x,y,nmodes=nmodes,nguess=1.82)
    # neffs,eexs,eeys = zhumodesolve(λ,n,x,y,nmodes=nmodes,nguess=1.72)
    neffs,eexs,eeys,*_ = zhumodesolve(λ,n,x,y,nmodes=nmodes,nguess=nguess,method=method,boundary=boundary)
    if polarization is not None:
        ns,Es = filtermodes(neffs,eexs,eeys,polarization=polarization)
        for i in range(len(ns)):
            # print(f"mode {i:2d}, effective index {ns[i]}")
            if plot: modeplot(Es[i],n,x,y)
        print(f"{len(ns)} {polarization} modes")
    else:
        for i in range(len(neffs)):
            print(f"mode {i:2d}, effective index {neffs[i]} polarization: {'x' if abs(eexs[i]).max()>abs(eeys[i]).max() else 'y'}")
            ee = eexs[i] if abs(eexs[i]).max()>abs(eeys[i]).max() else eeys[i]
            if plot: modeplot(ee,n,x,y)
def zhutest(λ=1.0,numx=81,numy=101,dn=0.01,plot=False):
    if 1:
        print('numx,numy',numx,numy)
        n,x,y = isotropicindex()
        neffs,eexs,eeys,*_ = zhumodesolve(λ,n,x,y,nmodes=10,nguess=None)
        print('neffs',neffs)
        if plot:
            modeplot(eexs[0],n,x,y)
            modeplot(eeys[0],n,x,y)
    if 1:
        n,x,y = anisotropicindex(dn,numx,numy)
        # plotindex(n,x,y,axis=1,clim=None,aspect=1)
        # neffs,eexs,eeys = zhumodesolve(λ,n,x,y,nmodes=1,nguess=2.5,boundary=('p','d'))
        # neffs,eexs,eeys = zhumodesolve(λ,n,x,y,nmodes=1,nguess=2.5,boundary=('d','p'))
        # neffs,eexs,eeys = zhumodesolve(λ,n,x,y,nmodes=1,nguess=2.5,boundary=('p','n'))
        # neffs,eexs,eeys = zhumodesolve(λ,n,x,y,nmodes=1,nguess=2.5,boundary=('n','p'))
        # neffs,eexs,eeys = zhumodesolve(λ,n,x,y,nmodes=1,nguess=2.5,boundary=('n','d'))
        # neffs,eexs,eeys = zhumodesolve(λ,n,x,y,nmodes=1,nguess=2.5,boundary=('d','n'))
        # neffs,eexs,eeys = zhumodesolve(λ,n,x,y,nmodes=1,nguess=2.5,boundary='p')
        # neffs,eexs,eeys = zhumodesolve(λ,n,x,y,nmodes=1,nguess=2.5,boundary='n')
        neffs,eexs,eeys,*_ = zhumodesolve(λ,n,x,y,nmodes=1,nguess=2.5,boundary='d')
        print('neffs',neffs)
        if plot:
            modeplot(eexs[0],n,x,y)
            modeplot(eeys[0],n,x,y)

if __name__ == '__main__':
    # cupytest()
    zhutest(plot=1)
    # zhutest(plot=1)
    # isotropicexample(plot=0)
    # anisotropicexample(nmodes=9,plot=0)
    # anisotropicexample(nmodes=20,polarization=None,plot=1)
    # anisotropicexample(nmodes=3,polarization='V',plot=1)
    # anisotropicexample(nmodes=40,polarization='H',plot=1)
    # anisotropicexample(nmodes=6,polarization='H',plot=1,method='teisotropic')
