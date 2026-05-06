
# modedata/modesolver overhaul
# x modedata args: λ,neffs,nn,Exs,Eys,Ezs,Hxs,Hys,Hzs,mode=None,nsub=None,pol=None,sell=None,tm=(True,False,None)
# x nn or nnx,nny,nnz?
# x optional Eys,Ezs,Hxs,Hys,Hzs?
# x move poynting vector to modedata
# x make polarization filtering optional and move to modedata
# x octavewgmodes return all fields
# x modesolve returns modedata from neffs,fields
# x decide whether to pass nx,ny,nz or εx,εy,εz
# x double check nan handling
# x define md[i]
# x define self.ee
# x define self.ees
# x define self.modenum
# x define self.dneff
# x define self.dneffs
# - zhusolve returns modedata from neffs,fields
# - figure out how to calculate dneff/dε from dneff/dn
# - modenum from modeid

# update tests from modesolve to solve
# newmodesolve → solve, ktpbendlossvsroccompare

# modesolve options:
# x octavesolve and zhusolve return raw modedata
# x def solve(self) takes care of polarization, mode selection, retries and returns curated modedata
# x nummodes
# x mode/targetmode
# x polarization
# x allmodes vs allmodes of a given polarization
# x method (exact, isotropic, supress)
# - retry at 3x nummodes if mode not found
# - solver (octavesolve, zhusolve)
# - boundary

import numpy as np
from numpy import pi,inf,sqrt,nan,sin,cos
from wavedata import Wave2D,Wave,wrange,tophat,inversestepfunction,storecallargs,maplistargs,lerp,track
from sellmeier import index,qpmwavelengths
import scipy,collections,joblib
from scipy.special import erf
# from modes import fibermode,Beamdata,pmfiber,gaussmode
memory = joblib.Memory('c:/work/modes/wgcache/',verbose=0)


class Modedata():
    def __init__(self,λ,neffs,nx,ny,nz,Exs,Eys,Ezs,Hxs,Hys,Hzs,modenum=None,pol=None,nsub=None,sell=None,wgargs={}):
        self.λ,self.neffs,self.nx,self.ny,self.nz = λ,neffs,nx,ny,nz
        self.Exs,self.Eys,self.Ezs,self.Hxs,self.Hys,self.Hzs = Exs,Eys,Ezs,Hxs,Hys,Hzs
        self.modenum,self.pol,self.nsub,self.sell,self.wgargs = modenum,pol,nsub,sell,wgargs
    @staticmethod
    def nan():
        return Modedata(nan,[nan],*(9*[np.array([[nan]])]),None,None,nan,'',{})
    def pols(self):
        def power(ee):
            return (np.abs(ee)**2).mean()
        def istm(eex,eey): # print(' '*9+f"powery/powerx:{power(eey)/power(eex):g} {'HV'[power(eex)<power(eey)]}")
            px,py = power(eex),power(eey)
            if np.isnan(px) or np.isnan(py): return 'nan'
            return power(eex)<power(eey)
        # return ''.join(['v' if istm(Ex,Ey) else 'h' for Ex,Ey in zip(self.Exs,self.Eys)])
        return ''.join([{True:'v',False:'h','nan':'-'}[istm(Ex,Ey)] for Ex,Ey in zip(self.Exs,self.Eys)])
    def filterpolarization(self,pol,verbose=False):
        assert pol in 'vh'
        neffs,Exs,Eys,Ezs,Hxs,Hys,Hzs = self.neffs,self.Exs,self.Eys,self.Ezs,self.Hxs,self.Hys,self.Hzs
        # neffs,Exs,Eys,Ezs,Hxs,Hys,Hzs = zip(*[(neff,Ex,Ey,Ez,Hx,Hy,Hz) for p,neff,Ex,Ey,Ez,Hx,Hy,Hz in zip(self.pols(),neffs,Exs,Eys,Ezs,Hxs,Hys,Hzs) if p==pol])
        filtered = [(neff,Ex,Ey,Ez,Hx,Hy,Hz) for p,neff,Ex,Ey,Ez,Hx,Hy,Hz in zip(self.pols(),neffs,Exs,Eys,Ezs,Hxs,Hys,Hzs) if p==pol or p=='-']
        neffs,Exs,Eys,Ezs,Hxs,Hys,Hzs = zip(*filtered) if filtered else ([],[],[],[],[],[],[])
        if verbose:
            print(f"filtered {len(self.neffs)} modes ({self.neffs[0]:g}..{self.neffs[-1]:g}) to {len(neffs)} modes {pol.upper()}:","".join([p.upper() for p in self.pols()]))
        return Modedata(self.λ,neffs,self.nx,self.ny,self.nz,Exs,Eys,Ezs,Hxs,Hys,Hzs,self.modenum,pol,self.nsub,self.sell,self.wgargs)
    def modecount(self):
        return len(self.neffs)
    def guidedmodecount(self,fractional=False):
        return Wave(1+np.arange(len(self.dneffs)),self.dneffs)(0) if fractional else len([n for n in self.dneffs if 0<=n])
    def modes(self):
        return [self[i] for i in range(self.modecount())]
    def guidedmodes(self):
        return [self[i] for i in range(self.guidedmodecount())]    
    def __getitem__(self,idx):
        idx,modenum = (idx,len(range(len(self.dneffs))[idx])-1) if isinstance(idx,slice) else (slice(idx,idx+1),0)
        # return Modedata(self.λ,self.neffs[idx],self.nx,self.ny,self.nz,self.Exs[idx],self.Eys[idx],self.Ezs[idx],self.Hxs[idx],self.Hys[idx],self.Hzs[idx],0,self.pol,self.nsub,self.sell,self.wgargs)
        # md = Modedata(λ=self.λ,neffs=self.neffs[idx],nx=self.nx,ny=self.ny,nz=self.nz,
        #                 Exs=self.Exs[idx],Eys=self.Eys[idx],Ezs=self.Ezs[idx],Hxs=self.Hxs[idx],Hys=self.Hys[idx],Hzs=self.Hzs[idx],
        #                 modenum=modenum,pol=self.pol,nsub=self.nsub,sell=self.sell,wgargs=self.wgargs)
        # print('md.modenum',md.modenum,'len',len(self.neffs),self.neffs[self.modenum])
        # return md
        return Modedata(λ=self.λ,neffs=self.neffs[idx],nx=self.nx,ny=self.ny,nz=self.nz,
                        Exs=self.Exs[idx],Eys=self.Eys[idx],Ezs=self.Ezs[idx],Hxs=self.Hxs[idx],Hys=self.Hys[idx],Hzs=self.Hzs[idx],
                        modenum=modenum,pol=self.pol,nsub=self.nsub,sell=self.sell,wgargs=self.wgargs)
    def __iter__(self):
        n,mds = 0,[self[i] for i in range(len(self.neffs))]
        while n<len(mds):
            yield mds[n]
            n += 1
    @property
    def neff(self): return self.neffs[self.modenum] if self.modenum is not None else np.nan
    @property
    def dneff(self): return self.neff-self.nsub
    @property
    def dneffs(self): return [n-self.nsub for n in self.neffs]
    @property
    def dnmax(self): return np.max(self.nn)-self.nsub
    @property
    def nn(self): return self.nx if 'h'==self.pol else self.ny if 'v'==self.pol else nan*self.nz
    @property
    def ees(self):
        Es = self.Exs if 'h'==self.pol else self.Eys if 'v'==self.pol else [nan*ee for ee in self.Ezs]
        def normalizesign(E): # make the biggest lobe positive, if it is left/right antisymmetric, make the left half positive
            with np.errstate(invalid='ignore'): # ignore nans in comparison
                ee = E[0:len(E.xs)//2,:]
                return E if np.iscomplexobj(ee) or -ee.min()<ee.max() else -E
        return [normalizesign(E) for E in Es]
    @property
    def ee(self):
        return self.ees[self.modenum].real() if self.modenum is not None else nan*self.nn
    @property
    def ii(self): return self.nn * self.ee.abs()**2 # I = ½ n c ε₀ E₀², E₀=amplitude
    @property
    def ex(self): return self.ee.atyindex(self.ee.maxindex()[1])
    @property
    def ey(self): return self.ee.atxindex(self.ee.maxindex()[0])
    @property
    def mfdx(self): return self.gaussianfit().mfdx
    @property
    def mfdy(self): return self.gaussianfit().mfdy
    def nax(self): return self.λ/pi/(self.mfdx*1000/2)
    def nay(self): return self.λ/pi/(self.mfdy*1000/2)
    def ellipticalmfdx(self): return self.ellipticaloverlap().mfdx
    def ellipticalmfdy(self): return self.ellipticaloverlap().mfdy
    def freespacemfdx(self): return self.freespaceoverlap.mfdx
    def freespacemfdy(self): return self.freespaceoverlap.mfdy
    def d4sigmax(self): return self.secondmoment().mfdx
    def d4sigmay(self): return self.secondmoment().mfdy
    def D86mfdx(self): return self.D86overlap().mfdx
    def D86mfdy(self): return self.D86overlap().mfdy
    def poyntingvector(self,component=None,i=None):
        # returns |S| if component not specified
        i = i if i is not None else self.modenum
        ex,ey,ez,hx,hy,hz = self.Exs[i],self.Eys[i],self.Ezs[i],self.Hxs[i],self.Hys[i],self.Hzs[i]
        sx,sy,sz = ey*hz-hy*ez, ez*hx-hz*ex, ex*hy-hx*ey
        return sx if 'x'==component else sy if 'y'==component else sz if 'z'==component else sqrt(sx**2+sy**2+sz**2)
    def spatialmodefft(self,units=None,nout=1,upsample=1): # nout = refractive index of output medium
        ee = self.ee.upsample(scale=upsample) # upsample increases maximum angle
        dx,dy,nx,ny = ee.dx,ee.dy,len(ee.xs),len(ee.ys)
        kx = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
        ky = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
        k0 = 1e3*2*pi*nout/self.λ
        c = 1/k0 if units=='radians' else 180/pi/k0 if units=='degrees' else 1
        return Wave2D(dx*dy*np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ee))), xs=c*kx, ys=c*ky)
    def spatialmodefftpeak(self):
        return [np.abs(k) for k in self.spatialmodefft().magsqr().xymax()]
    def angulardistributionx(self,units='degrees',nout=1):
        return self.spatialmodefft(units=units,nout=nout).abs().normalize().xslicemax()
    def angulardistributiony(self,units='degrees',nout=1):
        return self.spatialmodefft(units=units,nout=nout).abs().normalize().yslicemax()
    @property
    def Lc(self): # in mm
        return self.couplinglength(mode0=0,mode1=1)
    def couplinglength(self,mode0=0,mode1=1,symmetrywarning=True,atol=1e-4,plot=True): # in mm
        s,a = self.ees[mode0].issymmetric(atol=atol),self.ees[mode1].isasymmetric(atol=atol)
        if symmetrywarning:
            if not s:
                print(f'  Mode {mode0} is symmetric: {s}')
                if plot: 
                    self.ees[mode0].plot(legendtext=f'mode{mode0} not symmetric')
                    self.ees[mode1].plot(legendtext=f'mode{mode1}')
                    ww = self.ees[mode0]-self.ees[mode0].mirrorx()
                    Wave.plot( self.ees[mode0].xslice().rename('a'), self.ees[mode0].mirrorx().xslice().rename('b'), ww.xslice().rename('a-b, should be zero'), l='320',grid=1,legendtext=f'mode{mode0} not symmetric')
                return np.nan
            if not a:
                print(f'  Mode {mode1} is antisymmetric: {a}')
                if plot: 
                    self.ees[mode0].plot(legendtext=f'mode{mode0}')
                    self.ees[mode1].plot(legendtext=f'mode{mode1} not antisymmetric')
                    ww = self.ees[mode1]+self.ees[mode1].mirrorx()
                    Wave.plot( self.ees[mode1].xslice().rename('a'), self.ees[mode1].mirrorx().xslice().rename('b'), ww.xslice().rename('a+b, should be zero'), l='320',grid=1,legendtext=f'mode{mode1} not antisymmetric')
                return np.nan
            # if not (s and a): Wave.plot( *[self.ees[i].xslice().rename(f'mode{i}') for i in (mode0,mode1)] ,grid=1)
        Δn = self.neff - mode0.neff if isinstance(mode0,Modedata) else self.neffs[mode0] - self.neffs[mode1]
        return 1e-6*self.λ/np.abs(np.real(Δn))/2 # if np.real(Δn)>1e-8 else np.inf
        # µ = np.pi/2/Lc, T = sin²(L/Lc*π/2)
        # Leff = L + sqrt(pi/2*d0*R), µ=Aexp(d/d0) where T=sin²(µL)
    def crossover(self,L): # L in mm
        return np.sin(L/self.Lc*np.pi/2)**2
    def directionalcouplerkappa(self): # in 1/mm
        return np.pi/2/self.Lc # = 
    def directionalcouplerdelta(self,md): # in 1/mm # two modes are L and R
        return 1e6*np.pi*(self.neff-md.neff)/self.λ # δ = ½kΔneff = πΔneff/λ
    def directionalcouplercrossoveramplitude(self,z=None,Δneff=0): # z in mm, None for max crossover # Δneff = nleft - nright
        κ,δ = self.directionalcouplerkappa(),1e6*np.pi*Δneff/self.λ
        t = κ**2/(κ**2+δ**2)
        return t if z is None else t*np.sin(z*np.sqrt(κ**2+δ**2))
    def couplingphase(self,ys,xs,md=None,Lcwave=None): # in mm, ys = center-to-center splits, xs = propagation distances)
        if Lcwave is not None: # Lcwave = Wave(Lc,y)
            def µ(y):
                return np.pi/2 * 1/Lcwave(y,extrapolate='log')
            return np.trapz([µ(y) for y in ys],x=xs) # integrated phase over the propagation distance
        d0 = self.Lcfit(md) if md is not None else self.d0 # in um
        g0,L0 = self.args.split,self.Lc
        if len(xs)<2: return 0
        def µ(g): # Lc(g) = L0*exp((g-g0)/d0)
            L = L0*np.exp((g-g0)/d0)
            return np.pi/2 * 1/L
        return np.trapz([µ(y) for y in ys],x=xs) # integrated phase over the propagation distance
    def Lcfit(self,md,split=None,symmetrywarning=True): # in mm
        L1,L2 = self.couplinglength(symmetrywarning=symmetrywarning),md.couplinglength(symmetrywarning=symmetrywarning)
        g1,g2 = self.args.split,md.args.split
        # y1,y2 = np.log(L1),np.log(L2); m = (y2-y1)/(g2-g1); y0 = y1 - m*g1
        # Lc = Aexp(g/d0) → logLc = g/d0 + logA = m*g + y0 # d0 = 1/m
        d0 = (g2-g1)/(np.log(L2)-np.log(L1))
        self.d0 = d0
        return d0 if split is None else L1*np.exp((g-g1)/d0)
    def Lr(self,roc,md=None,symmetrywarning=True,leftrightsymmetric=False): # ROC in mm, Lr in mm
        # addition to effective Lc due to radius of curvature
        # includes both input and output s-bend, assuming one waveguide of the directional coupler has s-bends and the other waveguide is straight
        d0 = self.Lcfit(md,symmetrywarning=symmetrywarning) if md is not None else self.d0 # in um
        return 2*np.sqrt(pi/2*d0*roc*1000)/1000 * (1/sqrt(2) if leftrightsymmetric else 1)
    @staticmethod
    def arbitraryLr(Lcvssplit,splitvsz): # z in mm, Lc in mm, split in µm
        with np.errstate(over='ignore'):
            Lcvsz = Wave([Lcvssplit(split,extrapolate='log') for split in splitvsz.y],splitvsz.x)
        µvsz = 0.5*np.pi/Lcvsz
        ϕ = np.trapz(µvsz.y,x=µvsz.x) # integrated phase over the propagation distance
        Lc = Lcvssplit(splitvsz(0)) # assume min split at z=0
        return Lc*2/pi*ϕ

    def sbendLr(self,sx,sy,md=None,leftrightsymmetric=False,plot=False):
        from sellmeier import sbendroc,sbend
        g = self.args.split
        xs = np.linspace(-sx,sx,101)/1000
        ys = (2 if leftrightsymmetric else 1)*sbend(1000*np.abs(xs),sx,sy) + g
        # print(sbendroc(sx,sy),min(ys),max(ys),self.Lc*2/pi*self.couplingphase(ys,xs)/L1)
        if plot:
            wphase = Wave([self.couplingphase(ys[:n+1],xs[:n+1],md) for n in range(len(xs))],xs)
            wL,wy = self.Lc*2/pi*wphase,Wave(ys,xs)
            # Wave.plots(wphase,rightwaves=[wy],rightlabel='y (µm)',y='φ (rad)',x='x (mm)')
            Wave.plots(wL,rightwaves=[wy],rightlabel='y (µm)',y='Leff (mm)',x='x (mm)',
                save=f'sbend Leff {sx}sx {sy+g}pitch {g}split')
        return self.Lc*2/pi*self.couplingphase(ys,xs,md)
    def gaussianfit(self): return fitgaussian2D(self.ee) #,plotit=1)
    def modearea(self): return self.overlaparea(self,self)
    def apeprotondose(self): return self.cca.atx(0).area()
    def protondose(self): return self.ccr.atx(0).area()
    @property
    def args(self):
        class dotdict(dict): __getattr__,__setattr__,__delattr__ = dict.get,dict.__setitem__,dict.__delitem__
        return dotdict(self.wgargs)
    # def peakconcentration(self):
    #     return self.dnmax/self.dns # ccr.max()
    def overlap(self,md,plot=False,eemin=False):
        if plot or eemin:
            # P = ∫|E₀+E₁exp(iϕ)|²dσ = ∫ E̅₀E₀ + E̅₁E₁ + E̅₀E₁exp(iϕ) + E₀E̅₁exp(-iϕ) dσ
            # ∫ E̅₀E₀ dσ = P₀, ∫E̅₀E₁dσ = r exp(iθ), Pmin = P₀ + P₁ + 2r cos(θ+ϕ), ϕmin = π-θ
            # Emin = E₀+E₁exp(iϕmin) = E₀ - E₁exp(iθ), θ = arg(∫E̅₀E₁dσ)
            θ = np.angle(np.sum(self.ee.conj()*md.ee)) # print('θ:',θ)
            ee = self.ee - md.ee*np.exp(1j*θ)
            if plot:
                ee.magsqr().plot()
            if eemin:
                return ee            
        return self.ee.overlap(md.ee)
    
    # def overlaps(self,md,cs,L=0): # L in mm
    #     # self, md = new mode, old mode
    #     # md.ees = mode fields for old mode
    #     # cs = complex amplitude coefficients for each ee of md.ees
    #     # L = propagation length of old before coupling to new
    #     # returns new complex amplitude coefficients after coupling to new mode
    #     oldamplitudes = [cc*np.exp(1j*2*pi*nn/self.λ*L*1e6) for cc,nn in zip(cs,md.neffs)]
    #     # print(' --- ',oldamplitudes[:6])
    #     def newamplitude(e):
    #         return sum([ e.overlap(ee) * cc for ee,cc in zip(md.ees,cs) ])
    #     return [newamplitude(e) for e in self.ees]
    # def modesum(self,cs,plot=False):
    #     # for ee in self.ees: assert np.isrealobj(ee)
    #     # er = sum(ee*np.real(c) for ee,c in zip(self.ees,cs))
    #     # ei = sum(ee*np.imag(c) for ee,c in zip(self.ees,cs))
    #     # e = sqrt(er**2 + ei**2)
    #     ii = sum(ee*c for ee,c in zip(self.ees,cs)).magsqr()
    #     if plot:
    #         ii.plot()
    #     return ii
    def etchbraggcouplingconstant(self,dn,depth,width1,width0=0,dc=0.5,order=1): # in µm⁻¹
        def sinc(x): return np.sinc(x/np.pi)
        # assert np.allclose(np.sin(1)/1,sinc(1)) and np.allclose(np.sin(0.3)/0.3,sinc(0.3))
        gm = dc*np.abs(sinc(dc*order*np.pi)) # equivalent to np.sin(dc*order*np.pi)/(order*np.pi)
        return 2*np.pi*gm*dn*self.etchbraggoverlap(depth,width1,width0)/(1e-3*self.λ)
    def etchbraggoverlap(self,depth,width1,width0=0):
        ii = np.abs(self.ee)**2
        a = ii.sum()
        a1 = ii.subrange(xlim=(-0.5*width1,+0.5*width1),ylim=(-depth,0)).sum()
        a0 = ii.subrange(xlim=(-0.5*width0,+0.5*width0),ylim=(-depth,0)).sum()
        return (a1-a0)/a
    def couplingefficiency(self,md): # https://www.rp-photonics.com/mode_matching.html
        return self.overlap(md)
    def centroid(self):
        return collections.namedtuple('Position','x y')( np.sum(self.ee**2*self.ee.xx)/np.sum(self.ee**2), np.sum(self.ee**2*self.ee.yy)/np.sum(self.ee**2) )
    def secondmoment(self): # https://en.wikipedia.org/wiki/Beam_diameter#D4%CF%83_or_second-moment_width
        x0,y0 = self.centroid()
        ωx,ωy = 2*np.sqrt(np.sum(self.ee**2*(self.ee.xx-x0)**2)/np.sum(self.ee**2)), 2*np.sqrt(np.sum(self.ee**2*(self.ee.yy-y0)**2)/np.sum(self.ee**2))
        return Beamdata(1,ωx,ωy,x0,y0)
    def D86overlap(self): # https://en.wikipedia.org/wiki/Beam_diameter#D86_width
        def powerincircle(r,y0):
            return np.sum( self.ee**2 * step(r**2 - self.ee.xx**2 - (self.ee.yy-y0)**2, dx=self.ee.dx/10) ) / np.sum(self.ee**2)
        def f(x): # print(ω,y0,100*np.abs(powerincircle(ω,y0)))
            return 100*np.abs(powerincircle(*x)-0.86)**2 # + 1e-4*np.abs(ω) # have to minimize circle size too?
        ω,y0 = self.secondmoment().ω,self.centroid().y
        result = scipy.optimize.minimize(f, (ω,y0), options={'disp':False})
        ω,y0 = result.x
        return Beamdata(powerincircle(*result.x),ω,ω,x=0,y=y0)
    def freespaceoverlap(self):
        def f(x):
            ω,y0 = x
            return 1/self.gaussianoverlap(ωx=ω,ωy=ω,y0=y0)
        ω,y0 = self.secondmoment().ω,self.centroid().y
        result = scipy.optimize.minimize(f, (ω,y0), options={'disp':False})
        ω,y0 = result.x
        return Beamdata(1/result.fun,ω,ω,x=0,y=y0)
    def ellipticaloverlap(self):
        def f(x):
            ωx,ωy,y0 = x
            return 1/self.gaussianoverlap(ωx=ωx,ωy=ωy,y0=y0)
        ωx,ωy,y0 = self.secondmoment().ωx,self.secondmoment().ωy,self.centroid().y
        result = scipy.optimize.minimize(f, (ωx,ωy,y0), options={'disp':False})
        ωx,ωy,y0 = result.x
        return Beamdata(1/result.fun,ωx,ωy,x=0,y=y0)
    def fiberoverlap(self,returncontour=False,p0=None,fiber=None,θx=0,θy=0):
        if np.isnan(self.neff): return Beamdata(np.nan,np.nan,np.nan,np.nan,np.nan)
        def f(x):
            x0,y0 = x
            fd = fibermode(self.λ,x0=x0,y0=y0,res=(self.ee.dx,self.ee.dy),limits=self.ee.bounds(),fiber=fiber,θx=θx,θy=θy)
            return 1/self.overlap(fd)
        if p0 is not None:
            return 1/f(p0)
        x0,y0 = self.ee.xymax()
        result = scipy.optimize.minimize(f, (x0,y0), options={'disp':False})
        x0,y0 = result.x
        fd = fibermode(self.λ,x0=x0,y0=y0,res=self.ee.dx,limits=self.ee.bounds(),fiber=fiber,θx=θx,θy=θy)
        ωx,ωy = fd.secondmoment().ωx,fd.secondmoment().ωy
        if returncontour:
            theta = np.linspace(0,2*pi,101)
            ux,uy = x0+ωx*cos(theta),y0+ωy*sin(theta)
            return Wave(uy,ux)
        return Beamdata(1/result.fun,ωx,ωy,x0,y0)
    def fibercoupling(self,fiber=None):
        return self.fiberoverlap(fiber=fiber).amplitude
    def fiber(self):
        return pmfiber(self.λ)
    def pmfiber(self):
        return pmfiber(self.λ)
    def fibermode(self,x=None,y=None):
        bd = self.freespaceoverlap()
        x,y = x if x is not None else bd.x, y if y is not None else bd.y 
        return fibermode(self.λ,x0=x,y0=y,res=self.args.step,limits=self.args.limits)
    def gaussianoverlap(self,ωx,ωy=None,x0=0,y0=0,res=None,limits=None):
        res = res if res is not None else self.ee.dx
        limits = limits if limits is not None else self.ee.bounds()
        md = gaussmode(None,ωx,ωy,x0,y0,res,limits)
        return self.ee.overlap(md.ee)
    def deadenergy(self):
        if self.ee is None: return np.nan
        e = self.ee.copy()
        width,depth = self.args.width,self.args.depth # print('width,depth',width,depth)
        p0,p1 = e.x2p(-width/2),e.x2p(width/2)
        p0,p1 = int(np.ceil(p0)),int(np.floor(p1))+1
        q0,q1 = e.y2p(-depth),e.y2p(0)
        q0,q1 = int(np.ceil(q0)),int(np.floor(q1))+1
        e[p0:p1,q0:q1] = 0 # exchange area is dead
        return 1 - e.norm()/self.ee.norm()
    # def deadoverlaparea(self,md,md2):
    #     if self.ee is None: return np.nan
    #     e = self.ee.copy()
    #     e[:,:] = 1
    #     if not np.isclose( e.deadoverlap(self.ee,md.ee,md2.ee), self.ee.overlaparea(md.ee,md2.ee) ): print( 'Warning: failed np.isclose( e.deadoverlap(self.ee,md.ee,md2.ee), self.ee.overlaparea(md.ee,md2.ee)' )
    #     width,depth = self.args.width,self.args.depth # print('width,depth',width,depth)
    #     p0,p1 = e.x2p(-width/2),e.x2p(width/2)
    #     p0,p1 = int(np.ceil(p0)),int(np.floor(p1))+1
    #     q0,q1 = e.y2p(-depth),e.y2p(0)
    #     q0,q1 = int(np.ceil(q0)),int(np.floor(q1))+1
    #     a0 = e.deadoverlap(self.ee,md.ee,md2.ee)
    #     e[:,q1:] = 0 # air is dead
    #     a1 = e.deadoverlap(self.ee,md.ee,md2.ee)
    #     e[p0:p1,q0:q1] = 0 # exchange area is dead
    #     a2 = e.deadoverlap(self.ee,md.ee,md2.ee)
    #     # e[:p0,:] = 0 # outside channel is dead (no poling)
    #     # e[p1:,:] = 0
    #     # a3 = e.deadoverlap(self.ee,md.ee,md2.ee)
    #     # print('  ►► no dead:',a0,'µm²','air dead:',100*a0/a1,'%','exchange dead:',100*a0/a2,'%','outer dead:',100*a0/a3,'%','final:',a3,'µm²')
    #     return a2 # assume that bulk does get poled
    def overlaparea(self,md,md2):
        return self.ee.overlaparea(md.ee,md2.ee) if self.ee is not None else np.nan
    def modeidstr(self):
        ax,ay = self.modeid()
        return str(ax)+str(ay) if 0<=ax<10 and 0<=ay<10 else '?' #str(ax)+','+str(ay)
    # def modeids(self,asstring=False):
    #     # return [self.identifymode(k) for k in range(self.modecount())]
    #     return [self.identifymode(k,asstring=asstring) for k in range(len(self.neffs))]
    # def modeidnum(self,nx,ny=None):
    #     nx,ny = (nx,ny) if ny is not None else (tuple(int(x) for x in nx) if isinstance(nx,str) else nx)
    #     # for k in range(self.modecount()):
    #     for k in range(len(self.neffs)):
    #         if self.identifymode(k)==(nx,ny):
    #             return k
    #     return None
    def modeid(self,asstring=True,debug=False,plot2D=False):
        return self.identifymode(asstring=asstring,debug=debug,plot2D=plot2D)
    def identifymode(self,asstring=False,debug=False,plot2D=False):
        def lobes(ee,threshold=0.15): # returns [i,j] index of each lobe peak
            # mx,my = abs(ee).localmax(3,smoothsize=5)
            mx,my = abs(ee).localmax(3)
            return [(i,j) for i,j in zip(mx,my) if abs(ee[i,j])>threshold*abs(ee).max()] # lobe is noise (not counted) if height<threshold
        def lobesz(ee):
            return [abs(ee[x,y]) for x,y in lobes(ee)]
        def lobesxy(ee):
            return [(ee.xs[x],ee.ys[y]) for x,y in lobes(ee)]
        def triangulatelobes(ee):
            xys = lobesxy(ee)
            if len(xys)<2: return []
            if len(xys)==2: return [(0,1)]
            ps = [(x+1e-9*y**2,y+1e-9*x**2) for x,y in xys] # quadratic term added to avoid Delaunay colinearity error 
            def vertexpairs(points):
                from scipy.spatial import Delaunay
                tri = Delaunay(points) # print(tri.simplices,tri.vertex_neighbor_vertices,tri.npoints,tri.vertices)
                indices, indptr = tri.vertex_neighbor_vertices
                def neighbors(k): return indptr[indices[k]:indices[k+1]]
                return list(set([tuple(sorted((i,j))) for i in range(tri.npoints) for j in neighbors(i)]))
            return vertexpairs(ps)
        ee = self.ee
        if not (ee.issymmetric(horizontal=1) or ee.isasymmetric(horizontal=1)):
            print( np.max(np.abs(ee + ee.mirrorx())) , np.allclose(0*ee, ee + ee.mirrorx()) )
            print( np.max(np.abs(ee - ee.mirrorx())) , np.allclose(0*ee, ee - ee.mirrorx()) )
            # ee.plot()
        # assert ee.issymmetric(horizontal=1) or ee.isasymmetric(horizontal=1), 'lower default atol in issymmetric and isasymmetric (or asym waveguide not yet supported)'
        if not (ee.issymmetric(horizontal=1) or ee.isasymmetric(horizontal=1)): print('lower default atol in issymmetric and isasymmetric (or asym waveguide not yet supported)')

        if ee is None: return np.nan,np.nan
        def countbigpeaks1D(w):
            def ispeak(k):
                return w[k-1]<w[k] and w[k+1]<w[k]
            return sum( [1 for i in range(1,len(w)-1) if ispeak(i) and abs(w[i])>w.max()*1e-2 ] )
        nx,ny = countbigpeaks1D(abs(ee).ysum()),countbigpeaks1D(abs(ee).xsum()) # print(nx,ny)
        success = len(lobes(ee))==nx*ny
        if not success:
            # if modewarnings: print(f"mode couldn't be id'd from lobe positions, nx,ny={nx},{ny}, lobes={lobes(ee)}")
            nx,ny = np.nan,np.nan
            # plot = plot2D = True
        if debug: Wave.plots( abs(ee).ysum().rename('x'),abs(ee).xsum().rename('y'),xlabel='µm',pause=0 )
        if plot2D or debug:
            ec = ee.copy()
            for i,j in lobes(ec): ec[i,j] = 5
            ec.plot(pause=0)
        def tuple2string(ax,ay):
            return str(ax)+str(ay) if 0<=ax<10 and 0<=ay<10 else str(ax)+'-'+str(ay)
        return tuple2string(nx-1,ny-1) if asstring else (nx-1,ny-1)
    def fractionalpower(self,axis='x',plot=False):
        ii = self.poyntingvector('z').yaverage() if axis=='x' else self.poyntingvector('z').xaverage()
        u = ii.cumsum().normalize()
        if plot:
            u.plot(log=1,grid=1,x='x (µm)' if axis=='x' else 'y (µm)',y='fractional power to the left of x' if axis=='x' else 'fractional power below y')
        return u
    def bendloss(self,dbpermm=False,fit=True,plot=False):
        # MarcatiliMiller1969, Improved Relations Describing Directional Control in Electromagnetic Wave Guidance, p2164
        # xcrit = 1e3*r*(self.neff/self.nsub-1) # print(f'xcrit:{xcrit:g}µm r:{r:g}mm {self.ee.bounds()} {self.ee.xmin} {self.ee.xmax}')
        u = self.poyntingvector('z').yaverage().cumsum().normalize() # u.plot(log=1,x='x (µm)',y='fractional power to the left of x')
        w = Wave(u.y,1e-3*u.x/(self.neff/self.nsub-1)) # w.plot(log=1,x='R (mm)',y='fractional power to the left of xcrit')
        w = w(-inf,0)
        w = Wave(w.y,-w.x)
        a = self.d4sigmax() # x width of mode in µm # print(f"d4σx:{a:.2f}µm mfdx:{self.mfdx:.2f}µm")
        zc = 1e3 * 0.5 * a**2 * self.nsub/self.λ # power decay length in µm # print(f'power decay length Zc:{zc:.2f}µm')
        lossvsroc = 1e3 * 0.5/zc * w # mm⁻¹
        if fit:
            x0,x1 = w.xmax()/2,w.xmax()/2+w.dx()
            u = lerp(w.xwave,x0,x1,w.log()(x0),w.log()(x1)).exp() # Wave.plot(w,u,seed=1,log=1,grid=1,l='13')
            lossvsroc = 1e3 * 0.5/zc * u
        if plot:
            u = self.bendloss(dbpermm=dbpermm)
            C = self.weaklyguidinglossslope(dbpermm=dbpermm)
            uu = Wave(u(0)*10**(-0.05*C*u.x),u.x,l='3') if dbpermm else Wave(u(0)*np.exp(-C*u.x),u.x,l='3')
            Wave.plot(u,uu,xlim='f',grid=1,log=1,c='0',x='R (mm)',y='bend loss '+'(dB/mm)' if dbpermm else '(mm⁻¹)')
        # return lossvsroc # α as 1/e decay in mm⁻¹ vs R in mm
        return lossvsroc if not dbpermm else 20*np.log10(np.exp(1))*lossvsroc # dB/mm
    def weaklyguidinglossslope(self,dbpermm=False): # Vlasov04, Losses in single-mode silicon-on-insulator strip waveguides and bends, p1630
        # bend loss α ~ Kexp(-CR) where R is the bend radius in mm
        # α is the 1/e field decay length in mm⁻¹
        # power after length z = exp(-2αz) = exp(-2zKexp(-CR))
        Δn,n = self.dneff,self.neff # print(f'Δn,n,n0 {Δn:.4f} {n:.4f} {self.nsub:.4f}') 
        β = 2*pi*n/self.λ # in nm⁻¹
        C = 1e6*β*(2*Δn/n)**1.5 # in mm⁻¹
        return C if not dbpermm else 20*np.log10(np.exp(1))*C
    def lossslope(self,x=None,show=False): # Marcatili1969, Bends in Optical Dielectric Guides, p2115
        u = self.ee.xslicemax()
        x = x if x is not None else 0.5*self.ee.xmin
        ξ = 1/u.log().differentiate()(x) # field 1/e decay length in µm
        w = Wave(u(x)*np.exp((u.x-x)/ξ)+np.where(u.x<0,0,np.nan),u.x,f'{20*np.log10(np.exp(1))/ξ:g}dB/µm slope',c='k',l='1')
        if show:
            print(f'x,ξ = {x:.2f}µm,{ξ:.2f}µm')
            u.plot(w,grid=1,log=1,xlim=(None,0),x='x (µm)',y='relative field')
        β = 1e3*2*pi*self.neff/self.λ # in µm⁻¹
        C = (2/3) * β**2 / ξ**3 # in µm⁻¹
        print(f"warning, lossslope doesn't seem to be correct")
        print(f"β = {β:.2f}µm⁻¹")
        print(f"ξ = {ξ:.2f}µm⁻¹")
        print(f"C = {C:.2f}µm⁻¹")
        return 1e-3 * C # in mm⁻¹, bend loss α ~ Kexp(-CR) where R is the bend radius in mm
    def fitmode(self):
        bd = self.gaussianfit()
        return gaussmode(self.λ,bd.ωx,bd.ωy,bd.x,bd.y,res=self.args.res,limits=self.ee.bounds())
    def freespacemode(self):
        bd = self.freespaceoverlap()
        return gaussmode(self.λ,bd.ωx,bd.ωy,bd.x,bd.y,res=self.args.res,limits=self.ee.bounds())
    def ellipticalmode(self):
        bd = self.ellipticaloverlap()
        return gaussmode(self.λ,bd.ωx,bd.ωy,bd.x,bd.y,res=self.args.res,limits=self.ee.bounds())
    def plotxmodes(self,**kwargs):
        waves = [self.ex/self.ex(0),self.fitmode().ex,self.ellipticalmode().ex,self.freespacemode().ex,self.fibermode().ex]
        names = ['mode','fitmode','ellipticalmode','freespacemode','fibermode']
        Wave.plots(*[Wave(w,name=s) for w,s in zip(waves,names)],xlabel='µm',**kwargs)
    def plotymodes(self,**kwargs):
        waves = [self.ey/self.ey(0),self.fitmode().ey,self.ellipticalmode().ey,self.freespacemode().ey,self.fibermode().ey]
        names = ['mode','fitmode','ellipticalmode','freespacemode','fibermode']
        Wave.plots(*[Wave(w,name=s) for w,s in zip(waves,names)],xlabel='µm',**kwargs)
    def plotindex(self,*args,**kwargs):
        kwargs['x'] = kwargs.pop('x','µm')
        kwargs['y'] = kwargs.pop('y','µm')
        kwargs['colormap'] = kwargs.pop('colormap','inferno')
        vmin = kwargs.pop('vmin',self.nsub-0.01)
        return self.nn.plot(*args,vmin=vmin,contour=self.ee if kwargs.pop('contour',0) else None,**kwargs)
    def indexplot(self,*args,**kwargs):
        return self.plotindex(*args,**kwargs)
    def indexprofilex(self):
        return self.nn.xslicemax()
    def indexprofiley(self):
        return self.nn.atx(0)
    def plot(self,*args,**kwargs):
        kwargs['x'] = kwargs.pop('x','µm')
        kwargs['y'] = kwargs.pop('y','µm')
        self.ee.plot(*args,**kwargs)
        return self
    def plotmode(self,**kwargs):
        return self.plot(**kwargs)
    def modeplot(self,*args,**kwargs):
        return self.plot(*args,**kwargs)
    def saveinfo(self):
        self.plot(fewerticks=True,xlim=(-8,8),ylim=(-12,None),save=f'2D mode {self.λ}')
        self.ee.csv(f'2D mode {self.λ}')
        self.plotindex(fewerticks=True,xlim=(-8,8),ylim=(-12,None),save=f'2D index {self.λ}',contour=False)
        self.nn.csv(f'2D index {self.λ}')
        Wave.plots( (self.ex/self.ex.max()), rightwaves=[self.nn.xslicemax()],
            x='x (µm)',y='relative mode field',rightlabel='index',seed=0,
            ylim=(self.nsub-0.01,self.dnmax+self.nsub+0.01), save=f'horizontal profile {self.λ}' )
        Wave.plots( (self.ey/self.ey.max()), rightwaves=[self.nn.yslicemax()],
            x='y (µm)',y='relative mode field',rightlabel='index',seed=0,
            ylim=(self.nsub-0.01,self.dnmax+self.nsub+0.01), save=f'vertical profile {self.λ}' )
    def __eq__(self, other):
        return self.λ==other.λ
    def __lt__(self, other):
        return self.λ < other.λ
    def __repr__(self):
        return str(self)
    def __str__(self):
        return '  ► %4.0fλ %s mfdx%04.1f mfdy%04.1f Δn%5.3f' \
            % (self.λ,self.sell,self.mfdx,self.mfdy,self.dneff)

# modedata0 = 'λ,sell,neff,nn,nsub,ee,S,mode,pol,kwargs'.split(',') # wavelength, sellmeier, effective index, 2D index, substrate index, electric field, poynting vector, mode number, polarization, additional arguments
# Mdbase0 = collections.namedtuple('Modedata0', modedata0)
# class Modedata0(Mdbase0):
#     def __new__(cls, *args):
#         return super(Mdbase0,cls).__new__(cls, args)
#     @staticmethod
#     def nan():
#         return Modedata0(nan,'',[nan],nan,nan,[[nan]],[[nan]],0,'',{})
#     @property
#     def dneff(self): return self.neff-self.nsub
#     @property
#     def dnmax(self): return np.max(self.nn)-self.nsub
#     @property
#     def ii(self): # I = ½ n c ε₀ E₀², E₀=amplitude
#         return self.nn * self.ee.abs()**2
#     @property
#     def ex(self): return self.ee.atyindex(self.ee.maxindex()[1])
#     @property
#     def ey(self): return self.ee.atxindex(self.ee.maxindex()[0])
#     @property
#     def mfdx(self): return self.gaussianfit().mfdx
#     @property
#     def mfdy(self): return self.gaussianfit().mfdy
#     def efield(self): return self.ee
#     def poynting(self): return self.S
#     def nax(self): return self.λ/pi/(self.mfdx*1000/2)
#     def nay(self): return self.λ/pi/(self.mfdy*1000/2)
#     def ellipticalmfdx(self): return self.ellipticaloverlap().mfdx
#     def ellipticalmfdy(self): return self.ellipticaloverlap().mfdy
#     def freespacemfdx(self): return self.freespaceoverlap.mfdx
#     def freespacemfdy(self): return self.freespaceoverlap.mfdy
#     def d4sigmax(self): return self.secondmoment().mfdx
#     def d4sigmay(self): return self.secondmoment().mfdy
#     def D86mfdx(self): return self.D86overlap().mfdx
#     def D86mfdy(self): return self.D86overlap().mfdy
#     @property
#     def Lc(self): # in mm
#         correctmodes = self.ees[0].issymmetric() and self.ees[1].isasymmetric() # 
#         # print('correctmodes',correctmodes, self.ees[0].issymmetric(), self.ees[1].isasymmetric())
#         # return self.couplinglength(mode0=0,mode1=1) if correctmodes else np.nan
#         assert correctmodes, 'modes have wrong symmetry'
#         return self.couplinglength(mode0=0,mode1=1)
#     def directionalcouplerkappa(self): # in 1/mm
#         return np.pi/2/self.Lc
#     # def directionalcouplerdelta(self,Δneff): # in 1/mm
#     #     return 1e6*np.pi*Δneff/self.λ # δ = ½kΔneff = πΔneff/λ
#     def directionalcouplerdelta(self,md): # in 1/mm
#         return 1e6*np.pi*(self.neff-md.neff)/self.λ # δ = ½kΔneff = πΔneff/λ
#     def directionalcouplercrossoveramplitude(self,z=None,Δneff=0): # z=None for max crossover
#         κ,δ = self.directionalcouplerkappa(),1e6*np.pi*Δneff/self.λ
#         t = κ**2/(κ**2+δ**2)
#         return t if z is None else t*np.sin(z*np.sqrt(κ**2+δ**2))
#     def couplinglength(self,mode0=0,mode1=1): # in mm
#         # assert len(self.neffs)>1, 'need at least two modes'
#         Δn = self.neff - mode0.neff if isinstance(mode0,Modedata0) else self.neffs[mode0] - self.neffs[mode1]
#         return 1e-6*self.λ/np.abs(np.real(Δn))/2 # if np.real(Δn)>1e-8 else np.inf
#         # µ = np.pi/2/Lc, T = sin²(L/Lc*π/2)
#         # Leff = L + sqrt(pi/2*d0*R), µ=Aexp(d/d0) where T=sin²(µL)
#     def couplingphase(self,ys,xs,md=None): # in mm, ys = center-to-center gaps, xs = propagation distances)
#         d0 = self.Lcfit(md) if md is not None else self.d0 # in um
#         g0,L0 = self.args.gap,self.Lc
#         if len(xs)<2: return 0
#         def µ(g): # Lc(g) = L0*exp((g-g0)/d0)
#             L = L0*exp((g-g0)/d0)
#             return np.pi/2 * 1/L
#         µs = Wave([µ(y) for y in ys],xs)
#         return np.trapz(µs,x=xs) # integrated phase over the propagation distance
#     def Lcfit(self,md,gap=None): # in mm
#         L1,L2 = self.Lc,md.Lc
#         g1,g2 = self.args.gap,md.args.gap
#         # y1,y2 = np.log(L1),np.log(L2); m = (y2-y1)/(g2-g1); y0 = y1 - m*g1
#         # Lc = Aexp(g/d0) → logLc = g/d0 + logA = m*g + y0 # d0 = 1/m
#         d0 = (g2-g1)/(np.log(L2)-np.log(L1))
#         self.d0 = d0
#         return d0 if gap is None else L1*exp((g-g1)/d0)
#     def Lr(self,ROC,md=None): # ROC in mm, Lr in mm
#         # addition to effective Lc due to radius of curvature
#         # includes both input and output s-bend, assuming one waveguide of the directional coupler has s-bends and the other waveguide is straight
#         d0 = self.Lcfit(md) if md is not None else self.d0 # in um
#         return 2*np.sqrt(pi/2*d0*ROC*1000)/1000
#     def sbendLr(self,sx,sy,md=None,leftrightsymmetric=False,plot=False):
#         from sellmeier import sbendroc,sbend
#         g = self.args.gap
#         xs = np.linspace(-sx,sx,101)/1000
#         ys = (2 if leftrightsymmetric else 1)*sbend(1000*np.abs(xs),sx,sy) + g
#         # print(sbendroc(sx,sy),min(ys),max(ys),self.Lc*2/pi*self.couplingphase(ys,xs)/L1)
#         if plot:
#             wphase = Wave([self.couplingphase(ys[:n+1],xs[:n+1],md) for n in range(len(xs))],xs)
#             wL,wy = self.Lc*2/pi*wphase,Wave(ys,xs)
#             # Wave.plots(wphase,rightwaves=[wy],rightlabel='y (µm)',y='φ (rad)',x='x (mm)')
#             Wave.plots(wL,rightwaves=[wy],rightlabel='y (µm)',y='Leff (mm)',x='x (mm)',
#                 save=f'sbend Leff {sx}sx {sy+g}pitch {g}split')
#         return self.Lc*2/pi*self.couplingphase(ys,xs,md)
#     def gaussianfit(self): return fitgaussian2D(self.ee) #,plotit=1)
#     def modearea(self): return self.overlaparea(self,self)
#     def apeprotondose(self): return self.cca.atx(0).area()
#     def protondose(self): return self.ccr.atx(0).area()
#     @property
#     def args(self):
#         class dotdict(dict): __getattr__,__setattr__,__delattr__ = dict.get,dict.__setitem__,dict.__delitem__
#         return dotdict(self.kwargs)
#     def overlap(self,md):
#         return self.ee.overlap(md.ee)
#     def etchbraggcouplingconstant(self,dn,depth,width1,width0=0,dc=0.5,order=1): # in µm⁻¹
#         def sinc(x): return np.sinc(x/np.pi)
#         # assert np.allclose(np.sin(1)/1,sinc(1)) and np.allclose(np.sin(0.3)/0.3,sinc(0.3))
#         gm = dc*np.abs(sinc(dc*order*np.pi)) # equivalent to np.sin(dc*order*np.pi)/(order*np.pi)
#         return 2*np.pi*gm*dn*self.etchbraggoverlap(depth,width1,width0)/(1e-3*self.λ)
#     def etchbraggoverlap(self,depth,width1,width0=0):
#         ii = np.abs(self.ee)**2
#         a = ii.sum()
#         a1 = ii.subrange(xlim=(-0.5*width1,+0.5*width1),ylim=(-depth,0)).sum()
#         a0 = ii.subrange(xlim=(-0.5*width0,+0.5*width0),ylim=(-depth,0)).sum()
#         return (a1-a0)/a
#     def couplingefficiency(self,md): # https://www.rp-photonics.com/mode_matching.html
#         return self.overlap(md)
#     def centroid(self):
#         return collections.namedtuple('Position','x y')( np.sum(self.ee**2*self.ee.xx)/np.sum(self.ee**2), np.sum(self.ee**2*self.ee.yy)/np.sum(self.ee**2) )
#     def secondmoment(self): # https://en.wikipedia.org/wiki/Beam_diameter#D4%CF%83_or_second-moment_width
#         x0,y0 = self.centroid()
#         ωx,ωy = 2*np.sqrt(np.sum(self.ee**2*(self.ee.xx-x0)**2)/np.sum(self.ee**2)), 2*np.sqrt(np.sum(self.ee**2*(self.ee.yy-y0)**2)/np.sum(self.ee**2))
#         return Beamdata(1,ωx,ωy,x0,y0)
#     def D86overlap(self): # https://en.wikipedia.org/wiki/Beam_diameter#D86_width
#         def powerincircle(r,y0):
#             return np.sum( self.ee**2 * step(r**2 - self.ee.xx**2 - (self.ee.yy-y0)**2, dx=self.ee.dx/10) ) / np.sum(self.ee**2)
#         def f(x): # print(ω,y0,100*np.abs(powerincircle(ω,y0)))
#             return 100*np.abs(powerincircle(*x)-0.86)**2 # + 1e-4*np.abs(ω) # have to minimize circle size too?
#         ω,y0 = self.secondmoment().ω,self.centroid().y
#         result = scipy.optimize.minimize(f, (ω,y0), options={'disp':False})
#         ω,y0 = result.x
#         return Beamdata(powerincircle(*result.x),ω,ω,x=0,y=y0)
#     def freespaceoverlap(self):
#         def f(x):
#             ω,y0 = x
#             return 1/self.gaussianoverlap(ωx=ω,ωy=ω,y0=y0)
#         ω,y0 = self.secondmoment().ω,self.centroid().y
#         result = scipy.optimize.minimize(f, (ω,y0), options={'disp':False})
#         ω,y0 = result.x
#         return Beamdata(1/result.fun,ω,ω,x=0,y=y0)
#     def ellipticaloverlap(self):
#         def f(x):
#             ωx,ωy,y0 = x
#             return 1/self.gaussianoverlap(ωx=ωx,ωy=ωy,y0=y0)
#         ωx,ωy,y0 = self.secondmoment().ωx,self.secondmoment().ωy,self.centroid().y
#         result = scipy.optimize.minimize(f, (ωx,ωy,y0), options={'disp':False})
#         ωx,ωy,y0 = result.x
#         return Beamdata(1/result.fun,ωx,ωy,x=0,y=y0)
#     def fiberoverlap(self,returncontour=False,p0=None,fiber=None,θx=0,θy=0):
#         if np.isnan(self.neff): return Beamdata(np.nan,np.nan,np.nan,np.nan,np.nan)
#         def f(x):
#             x0,y0 = x
#             fd = fibermode(self.λ,x0=x0,y0=y0,res=(self.ee.dx,self.ee.dy),limits=self.ee.bounds(),fiber=fiber,θx=θx,θy=θy)
#             return 1/self.overlap(fd)
#         if p0 is not None:
#             return 1/f(p0)
#         x0,y0 = self.ee.xymax()
#         result = scipy.optimize.minimize(f, (x0,y0), options={'disp':False})
#         x0,y0 = result.x
#         fd = fibermode(self.λ,x0=x0,y0=y0,res=self.ee.dx,limits=self.ee.bounds(),fiber=fiber,θx=θx,θy=θy)
#         ωx,ωy = fd.secondmoment().ωx,fd.secondmoment().ωy
#         if returncontour:
#             theta = np.linspace(0,2*pi,101)
#             ux,uy = x0+ωx*cos(theta),y0+ωy*sin(theta)
#             return Wave(uy,ux)
#         return Beamdata(1/result.fun,ωx,ωy,x0,y0)
#     def fibercoupling(self,fiber=None):
#         return self.fiberoverlap(fiber=fiber).amplitude
#     def fiber(self):
#         return pmfiber(self.λ)
#     def pmfiber(self):
#         return pmfiber(self.λ)
#     def fibermode(self,x=None,y=None):
#         bd = self.freespaceoverlap()
#         x,y = x if x is not None else bd.x, y if y is not None else bd.y 
#         return fibermode(self.λ,x0=x,y0=y,res=self.args.res,limits=self.args.limits)
#     def gaussianoverlap(self,ωx,ωy=None,x0=0,y0=0,res=None,limits=None):
#         res = res if res is not None else self.ee.dx
#         limits = limits if limits is not None else self.ee.bounds()
#         md = gaussmode(None,ωx,ωy,x0,y0,res,limits)
#         return self.ee.overlap(md.ee)
#     def deadenergy(self):
#         if self.ee is None: return np.nan
#         e = self.ee.copy()
#         width,depth = self.args.width,self.args.depth # print('width,depth',width,depth)
#         p0,p1 = e.x2p(-width/2),e.x2p(width/2)
#         p0,p1 = int(np.ceil(p0)),int(np.floor(p1))+1
#         q0,q1 = e.y2p(-depth),e.y2p(0)
#         q0,q1 = int(np.ceil(q0)),int(np.floor(q1))+1
#         e[p0:p1,q0:q1] = 0 # exchange area is dead
#         return 1 - e.norm()/self.ee.norm()
#     def overlaparea(self,md,md2):
#         return self.ee.overlaparea(md.ee,md2.ee) if self.ee is not None else np.nan
#     def crossover(self,L): # L in mm
#         return np.sin(L/self.Lc*np.pi/2)**2
#     def modeidstr(self):
#         ax,ay = self.modeid
#         return str(ax)+str(ay) if 0<=ax<10 and 0<=ay<10 else '?' #str(ax)+','+str(ay)
#     def modeid(self,asstring=False,debug=False,plot2D=False):
#         return self.identifymode(asstring=asstring,debug=debug,plot2D=plot2D)
#     def identifymode(self,asstring=False,debug=False,plot2D=False):
#         def lobes(ee,threshold=0.15): # returns [i,j] index of each lobe peak
#             # mx,my = abs(ee).localmax(3,smoothsize=5)
#             mx,my = abs(ee).localmax(3)
#             return [(i,j) for i,j in zip(mx,my) if abs(ee[i,j])>threshold*abs(ee).max()] # lobe is noise (not counted) if height<threshold
#         def lobesz(ee):
#             return [abs(ee[x,y]) for x,y in lobes(ee)]
#         def lobesxy(ee):
#             return [(ee.xs[x],ee.ys[y]) for x,y in lobes(ee)]
#         def triangulatelobes(ee):
#             xys = lobesxy(ee)
#             if len(xys)<2: return []
#             if len(xys)==2: return [(0,1)]
#             ps = [(x+1e-9*y**2,y+1e-9*x**2) for x,y in xys] # quadratic term added to avoid Delaunay colinearity error 
#             def vertexpairs(points):
#                 from scipy.spatial import Delaunay
#                 tri = Delaunay(points) # print(tri.simplices,tri.vertex_neighbor_vertices,tri.npoints,tri.vertices)
#                 indices, indptr = tri.vertex_neighbor_vertices
#                 def neighbors(k): return indptr[indices[k]:indices[k+1]]
#                 return list(set([tuple(sorted((i,j))) for i in range(tri.npoints) for j in neighbors(i)]))
#             return vertexpairs(ps)
#         ee = self.ee
#         if not (ee.issymmetric(horizontal=1) or ee.isasymmetric(horizontal=1)):
#             print( np.max(np.abs(ee + ee.mirrorx())) , np.allclose(0*ee, ee + ee.mirrorx()) )
#             print( np.max(np.abs(ee - ee.mirrorx())) , np.allclose(0*ee, ee - ee.mirrorx()) )
#             # ee.plot()
#         # assert ee.issymmetric(horizontal=1) or ee.isasymmetric(horizontal=1), 'lower default atol in issymmetric and isasymmetric (or asym waveguide not yet supported)'
#         if not (ee.issymmetric(horizontal=1) or ee.isasymmetric(horizontal=1)): print('lower default atol in issymmetric and isasymmetric (or asym waveguide not yet supported)')

#         if ee is None: return np.nan,np.nan
#         def countbigpeaks1D(w):
#             def ispeak(k):
#                 return w[k-1]<w[k] and w[k+1]<w[k]
#             return sum( [1 for i in range(1,len(w)-1) if ispeak(i) and abs(w[i])>w.max()*1e-2 ] )
#         nx,ny = countbigpeaks1D(abs(ee).ysum()),countbigpeaks1D(abs(ee).xsum()) # print(nx,ny)
#         success = len(lobes(ee))==nx*ny
#         if not success:
#             # if modewarnings: print(f"mode couldn't be id'd from lobe positions, nx,ny={nx},{ny}, lobes={lobes(ee)}")
#             nx,ny = np.nan,np.nan
#             # plot = plot2D = True
#         if debug: Wave.plots( abs(ee).ysum().rename('x'),abs(ee).xsum().rename('y'),xlabel='µm',pause=0 )
#         if plot2D or debug:
#             ec = ee.copy()
#             for i,j in lobes(ec): ec[i,j] = 5
#             ec.plot(pause=0)
#         def tuple2string(ax,ay):
#             return str(ax)+str(ay) if 0<=ax<10 and 0<=ay<10 else str(ax)+'-'+str(ay)
#         return tuple2string(nx-1,ny-1) if asstring else (nx-1,ny-1)
#     def bendloss(self,dbpermm=False,fit=True):
#         # MarcatiliMiller1969, Improved Relations Describing Directional Control in Electromagnetic Wave Guidance, p2164
#         # xcrit = 1e3*r*(self.neff/self.nsub-1) # print(f'xcrit:{xcrit:g}µm r:{r:g}mm {self.ee.bounds()} {self.ee.xmin} {self.ee.xmax}')
#         u = self.S.yaverage().cumsum().normalize() # u.plot(log=1,x='x (µm)',y='fractional power to the left of x')
#         w = Wave(u.y,1e-3*u.x/(self.neff/self.nsub-1)) # w.plot(log=1,x='R (mm)',y='fractional power to the left of xcrit')
#         w = w(-inf,0)
#         w = Wave(w.y,-w.x)
#         a = self.d4sigmax() # x width of mode in µm # print(f"d4σx:{a:.2f}µm mfdx:{self.mfdx:.2f}µm")
#         zc = 1e3 * 0.5 * a**2 * self.nsub/self.λ # power decay length in µm # print(f'power decay length Zc:{zc:.2f}µm')
#         lossvsroc = 1e3 * 0.5/zc * w # mm⁻¹
#         if fit:
#             x0,x1 = w.xmax()/2,w.xmax()/2+w.dx()
#             u = lerp(w.xwave,x0,x1,w.log()(x0),w.log()(x1)).exp() # Wave.plot(w,u,seed=1,log=1,grid=1,l='13')
#             lossvsroc = 1e3 * 0.5/zc * u
#         # return lossvsroc # α as 1/e decay in mm⁻¹ vs R in mm
#         return lossvsroc if not dbpermm else 20*np.log10(np.exp(1))*lossvsroc # dB/mm
#     def weaklyguidinglossslope(self): # Vlasov04, Losses in single-mode silicon-on-insulator strip waveguides and bends, p1630
#         # bend loss α ~ Kexp(-CR) where R is the bend radius in mm
#         # α is the 1/e field decay length in mm⁻¹
#         # power after length z = exp(-2αz) = exp(-2zKexp(-CR))
#         Δn,n = self.dneff,self.neff # print(f'Δn,n,n0 {Δn:.4f} {n:.4f} {self.nsub:.4f}') 
#         β = 2*pi*n/self.λ # in nm⁻¹
#         C = 1e6*β*(2*Δn/n)**1.5 # in mm⁻¹
#         return C
#     def lossslope(self,x=None,show=False): # Marcatili1969, Bends in Optical Dielectric Guides, p2115
#         u = self.ee.xslicemax()
#         x = x if x is not None else 0.5*self.ee.xmin
#         ξ = 1/u.log().differentiate()(x) # field 1/e decay length in µm
#         w = Wave(u(x)*np.exp((u.x-x)/ξ)+np.where(u.x<0,0,np.nan),u.x,f'{20*np.log10(np.exp(1))/ξ:g}dB/µm slope',c='k',l='1')
#         if show:
#             print(f'x,ξ = {x:.2f}µm,{ξ:.2f}µm')
#             u.plot(w,grid=1,log=1,xlim=(None,0),x='x (µm)',y='relative field')
#         β = 1e3*2*pi*self.neff/self.λ # in µm⁻¹
#         C = (2/3) * β**2 / ξ**3 # in µm⁻¹
#         print(f"β = {β:.2f}µm⁻¹")
#         print(f"ξ = {ξ:.2f}µm⁻¹")
#         print(f"C = {C:.2f}µm⁻¹")
#         print(f"warning, doesn't seem to be correct")
#         return 1e-3 * C # in mm⁻¹, bend loss α ~ Kexp(-CR) where R is the bend radius in mm
#     def fitmode(self):
#         bd = self.gaussianfit()
#         return gaussmode(self.λ,bd.ωx,bd.ωy,bd.x,bd.y,res=self.args.res,limits=self.ee.bounds())
#     def freespacemode(self):
#         bd = self.freespaceoverlap()
#         return gaussmode(self.λ,bd.ωx,bd.ωy,bd.x,bd.y,res=self.args.res,limits=self.ee.bounds())
#     def ellipticalmode(self):
#         bd = self.ellipticaloverlap()
#         return gaussmode(self.λ,bd.ωx,bd.ωy,bd.x,bd.y,res=self.args.res,limits=self.ee.bounds())
#     def plotxmodes(self,**kwargs):
#         waves = [self.ex/self.ex(0),self.fitmode().ex,self.ellipticalmode().ex,self.freespacemode().ex,self.fibermode().ex]
#         names = ['mode','fitmode','ellipticalmode','freespacemode','fibermode']
#         Wave.plots(*[Wave(w,name=s) for w,s in zip(waves,names)],xlabel='µm',**kwargs)
#     def plotymodes(self,**kwargs):
#         waves = [self.ey/self.ey(0),self.fitmode().ey,self.ellipticalmode().ey,self.freespacemode().ey,self.fibermode().ey]
#         names = ['mode','fitmode','ellipticalmode','freespacemode','fibermode']
#         Wave.plots(*[Wave(w,name=s) for w,s in zip(waves,names)],xlabel='µm',**kwargs)
#     def plotindex(self,*args,**kwargs):
#         kwargs['x'] = kwargs.pop('x','µm')
#         kwargs['y'] = kwargs.pop('y','µm')
#         kwargs['colormap'] = kwargs.pop('colormap','inferno')
#         return self.nn.plot(*args,vmin=self.nsub-0,contour=self.ee if kwargs.pop('contour',0) else None,**kwargs)
#         # from wavedata import plot
#         # return plot(image=self.nn,contour=self.ee if kwargs.pop('contour',0) else None,**kwargs)
#     def indexplot(self,*args,**kwargs):
#         return self.plotindex(*args,**kwargs)
#     def indexprofilex(self):
#         return self.nn.xslicemax()
#     def indexprofiley(self):
#         return self.nn.atx(0)
#     def plot(self,*args,**kwargs):
#         kwargs['x'] = kwargs.pop('x','µm')
#         kwargs['y'] = kwargs.pop('y','µm')
#         self.ee.plot(*args,**kwargs)
#         return self
#     def saveinfo(self):
#         self.plot(fewerticks=True,xlim=(-8,8),ylim=(-12,None),save=f'2D mode {self.λ}')
#         self.ee.csv(f'2D mode {self.λ}')
#         self.plotindex(fewerticks=True,xlim=(-8,8),ylim=(-12,None),save=f'2D index {self.λ}',contour=False)
#         self.nn.csv(f'2D index {self.λ}')
#         Wave.plots( (self.ex/self.ex.max()), rightwaves=[self.nn.xslicemax()],
#             x='x (µm)',y='relative mode field',rightlabel='index',seed=0,
#             ylim=(self.nsub-0.01,self.dnmax+self.nsub+0.01), save=f'horizontal profile {self.λ}' )
#         Wave.plots( (self.ey/self.ey.max()), rightwaves=[self.nn.yslicemax()],
#             x='y (µm)',y='relative mode field',rightlabel='index',seed=0,
#             ylim=(self.nsub-0.01,self.dnmax+self.nsub+0.01), save=f'vertical profile {self.λ}' )
#     def __eq__(self, other):
#         return self.λ==other.λ
#     def __lt__(self, other):
#         return self.λ < other.λ
#     def __repr__(self):
#         return str(self)
#     def __str__(self):
#         return '  ► %4.0fλ %s mfdx%04.1f mfdy%04.1f Δn%5.3f' \
#             % (self.λ,self.sell,self.mfdx,self.mfdy,self.dneff)

from dataclasses import dataclass, field
@dataclass
class Qpmdata:
    md1: Modedata
    md2: Modedata
    md3: Modedata
    args: dict = field(default_factory=dict)
    @property
    def Λ(self):
        Λ0 = 1/(self.md3.neff/self.λ3 - self.md2.neff/self.λ2 - self.md1.neff/self.λ1)/1000
        return Λ0 if not np.iscomplex(Λ0) else np.real(Λ0) # if np.isclose(np.imag(Λ0),0,atol=1e-2) else Λ0
    # @property
    # def Λbulk(self): return 1/(self.md3.nsub/self.λ3 - self.md2.nsub/self.λ2 - self.md1.nsub/self.λ1)/1000
    @property
    def λ1(self): return self.md1.λ
    @property
    def λ2(self): return self.md2.λ
    @property
    def λ3(self): return self.md3.λ
    @property
    def sell(self):
        if not self.md1.sell:
            return self.md1.bulk
        return self.md1.sell[:-1]
    @property
    def Type(self):
        if not self.md1.sell:
            assert 0
            return ''.join('yz'[md.tm] for md in (self.md1,self.md2,self.md3))
        return ''.join([md.sell[-1] for md in (self.md1,self.md2,self.md3)])
    # def deff(self): return {'zzz':{'ktp':9.8,'fan':9.8,'ln':27*2/np.pi,'lnridge':27*2/np.pi,'mgln':16,'mglnridge':16,'slt':9},
    #                         'yzy':{'ktp':2.4,'fan':2.4,'ln':2.8,'lnridge':2.8,'mgln':2.8,'mglnridge':2.8},
    #                         'yyz':{'ktp':2.4,'fan':2.4}}[self.Type][self.sell]
    def deff(self):
        from sellmeier import dij as sellmeierdij
        # print(f'warning, using new deff value {sellmeierdij(sell=self.sell,Type=self.Type):g}pm/V for {self.sell} {self.Type}')
        return 2/pi*sellmeierdij(sell=self.sell,Type=self.Type)
    def overlaparea(self,dmask=None):
        dmask = self.args['dmask'] if dmask is None and 'dmask' in self.args else dmask
        # if 'dmask' in self.args:
        #     print('self.args.dmask',self.args['dmask'],'dmask',dmask)
        if dmask is None:
            return self.md1.overlaparea(self.md2,self.md3)
        return (dmask+0*self.md1.ee).deadoverlap(self.md1.ee,self.md2.ee,self.md3.ee)

        # if dmask is not None:
        #     return self.md1.overlaparea(self.md2,self.md3) if 1==dmask else (dmask+0*self.md1.ee).deadoverlap(self.md1.ee,self.md2.ee,self.md3.ee)
        # if 'dmask' in self.args:
        #     return (dmask+0*self.md1.ee).deadoverlap(self.md1.ee,self.md2.ee,self.md3.ee)
        # return self.md1.overlaparea(self.md2,self.md3)
    def ce(self):
        # if self.md1.sell in ['ln','mgln','lnz','mglnz','lny','mglny']:
        #     print(f'warning: dead layer not considered for conversion efficiency, sell={self.md1.sell}')
        return self.conversionefficiency() # shg equivalent conversion efficiency in %/W/cm²
    def shgce(self): return self.ce()
    def sfgce(self): return self.ce()*4
    def npshgce(self): return self.ce()*(np.pi**2/4) # useful although type II SHG nonpoled is technically SFG
    def npsfgce(self): return self.sfgce()*(np.pi**2/4)
    # def deadce(self): return self.conversionefficiency(dead=True)
    # def deadshgce(self): return self.deadce()
    # def deadsfgce(self): return self.deadce()*4
    def mds(self): return [self.md1,self.md2,self.md3]
    def __eq__(self, other):
        return (self.md1,self.md2,self.md3)==(other.md1,other.md2,other.md3)
    def __lt__(self, other):
        return (self.md1,self.md2,self.md3) < (other.md1,other.md2,other.md3)
    # def __str__(self):
    #     try:
    #         return str(self.md1.nsub)+' '+str(index(self.λ1,self.md1.sell,20)) # fix index does not agree
    #     except:
    #         return str(self.md1.nsub)
    def invΛ(self):
        return 1/self.Λ
    def conversionefficiency(self,dmask=None):
        area = self.overlaparea(dmask=dmask)
        #  32e5 * 2.99792458 * pi**3 * deff**2 / λ1 / λ2 / nsub1 / nsub2 / nsub3 / area
        return 100 * 8*np.pi**2*40*np.pi*2.99792458*1e2 * self.deff()**2 /self.λ1/self.λ2 /self.md1.neff/self.md2.neff/self.md3.neff/area
    def photonconversionP1(self,L): # L in mm, power in W at λ1 for 100% photon conversion of λ2 photons to λsfg # P1 = (λ2/λSFG) π²/(4 ηSFG)
        ηsfg = 4*self.conversionefficiency()/100 * (L/10)**2
        return np.pi**2/4 * self.λ2/self.λ3 / ηsfg
    def photonconversionP2(self,L):
        return self.photonconversionP1(L) * self.λ1/self.λ2
    def Λtemp(self,T):
        from sellmeier import polingperiod
        Λ0 = polingperiod(w1=self.λ1,w2=self.λ2,temp=20,sell=self.sell,Type=self.Type)
        ΛT = polingperiod(w1=self.λ1,w2=self.λ2,temp= T,sell=self.sell,Type=self.Type)
        return 1/(1/self.Λ - 1/Λ0 + 1/ΛT)
    def phasematchingangle(self,d=1):
        Λ0 = self.Λ
        args = self.args.copy()
        λ1,λ2 = args.pop('λ1'),args.pop('λ2')
        λ1s = [λ1,λ1+d] # np.linspace(λ1-d,λ1+d,2)
        # wp = Wave([1/qpm(λ1=λ,λ2=λ2+d,**args).Λ - 1/Λ0 for λ in λ1s],λ1s)
        # wn = Wave([1/qpm(λ1=λ,λ2=λ2-d,**args).Λ - 1/Λ0 for λ in λ1s],λ1s)
        # def u(w): return Wave([w[0],w[-1]],[w.x[0],w.x[-1]])
        # Wave.plots(wp,u(wp),l='03',m='o ',grid=1,fewerticks=1)
        # Wave.plots(wp,wn,u(wp),u(wn),l='0033',m='oo  ',grid=1,fewerticks=1)
        # d1 = wp.xaty(0) - λ2
        w = Wave(λ1s,[1/qpm(λ1=λ,λ2=λ2+d,**args).Λ - 1/Λ0 for λ in λ1s])
        d1 = w.atx(0,extrapolate='lin') - λ1
        return np.arctan2(d1,d)*180/pi # np.arctan2(y,x)
    def pairrate(self,L):
        from sellmeier import pairrate
        print(self.sell,self.Type)
        print(f"Λ={self.Λ}")
        print(f"Λ={polingperiod(self.λ1,self.λ2,sell=self.sell+'wg',Type=self.Type)}")
        ce = self.shgce
        print(f"η = {ce:.1f}%/W/cm²")
        print(pairrate(self.λ1,self.λ2,sell=self.sell+'wg',Type=self.Type,L=L,ηSHG=ce,nW=False,units=False)/1e9,'GHz/mW')
    def plot(self):
        self.md1.plot()
        self.md2.plot()
        self.md3.plot()
    def props():
        return 'Λ,Λbulk,overlaparea,λ1,λ2,λ3,deff,ce'.split(',')

@memory.cache
def octavesolver(λ,epsx,epsy,epsz,nummodes,dx,dy,boundary,method='exact',nguess=None,verbose=False):
    # https://www.photonics.umd.edu/software/wgmodes/ # A. B. Fallahkhair, K. S. Li and T. E. Murphy, "Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides", J. Lightwave Technol. 26(11), 1423-1431, (2008).
    # boundary: 4 letter string specifying boundary conditions to be applied at the edges of the computation window. [North,South,East,West] 'A' - Hx is antisymmetric, Hy is symmetric. 'S' - Hx is symmetric and, Hy is antisymmetric. '0' - Hx and Hy are zero immediately outside of the boundary.
    import os,oct2py
    os.environ["OCTAVE_EXECUTABLE"] = r"C:\octave\Octave-4.4.1\bin\octave-cli.exe"
    oct2py.octave.addpath('oct')
    assert method in ('exact','isotropic','supress')
    if 'isotropic'==method:
        epsguess = np.max(epsx) if nguess is None else nguess**2
        hx,hy,neff = oct2py.octave.wgmodes(λ/1000, epsguess, nummodes, dx, dy, epsx, boundary, nout=3)
    else:
        epsguess = max(np.max(epsx),np.max(epsy))
        hx,hy,neff = oct2py.octave.wgmodes(λ/1000, epsguess, nummodes, dx, dy, epsx, epsy, epsz, boundary, nout=3)
    if 3==len(hx.shape):    # more than one mode
        neffs,hxs,hys = [ni[0] for ni in neff],hx[:,:,:],hy[:,:,:]
    else:                   # one mode
        neffs,hxs,hys = [neff],hx[:,:,np.newaxis],hy[:,:,np.newaxis]
    def fields(neff,hhx,hhy): # note: hhz,eex,eey,eez are labeled according to wgmodes axes
        if 'isotropic'==method:
            hhz,eex,eey,eez = oct2py.octave.postprocess(λ/1000, neff, hhx, hhy, dx, dy, epsx, boundary, nout=4)
        else:
            hhz,eex,eey,eez = oct2py.octave.postprocess(λ/1000, neff, hhx, hhy, dx, dy, epsx, epsy, epsz, boundary, nout=4)
        return hhz,eex,eey,eez
    hxs,hys = [hxs[:,:,i] for i in range(len(neffs))],[hys[:,:,i] for i in range(len(neffs))]
    hzs,exs,eys,ezs = zip(*[fields(neffs,hx,hy) for neffs,hx,hy in zip(neffs,hxs,hys)])
    return neffs,exs,eys,ezs,hxs,hys,hzs

# @memory.cache
def newoctavewgmodes(λ,epsxx,epsyy,epszz,nummodes,dx,dy,boundary,isotropic=False,exact=False,tm=True,verbose=False,targetmode=None):
    # https://www.photonics.umd.edu/software/wgmodes/ # A. B. Fallahkhair, K. S. Li and T. E. Murphy, "Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides", J. Lightwave Technol. 26(11), 1423-1431, (2008).
    # boundary: 4 letter string specifying boundary conditions to be applied at the edges of the computation window. [North,South,East,West] 'A' - Hx is antisymmetric, Hy is symmetric. 'S' - Hx is symmetric and, Hy is antisymmetric. '0' - Hx and Hy are zero immediately outside of the boundary.
    import os,oct2py
    os.environ["OCTAVE_EXECUTABLE"] = r"C:\octave\Octave-4.4.1\bin\octave-cli.exe"
    oct2py.octave.addpath('oct')
    assert (not isotropic and not exact) or targetmode is not None, 'must specify targetmode for exact or isotropic'
    targetmode = targetmode if targetmode is not None else nummodes-1
    def modifyeps(epsxx,epsyy,δn=0.1):
        # modesolver can't find the solution when desired polarization is lower index than other axis, so modify the other axis to be lower index. this is not done for 'exact' or 'isotropic'
        # for 'exact', we find all solutions and filter out wrong polarization (of which all but the last few are)
        # for 'isotropic', both polarizations have roughly the same neff, so we find twice as many modes and filter out half
        def modified(eps,dn):
            return (sqrt(eps)-dn)**2
        nx,ny = [np.max(sqrt(eps)) for eps in (epsxx,epsyy)]
        if tm and ny<nx+δn:
            return modified(epsxx,nx+δn-ny),epsyy
        elif not tm and nx<ny+δn:
            return epsxx,modified(epsyy,ny+δn-nx)
        else:
            return epsxx,epsyy
    if not isotropic and not exact:
        epsxx,epsyy = modifyeps(epsxx,epsyy)
    if isotropic:
        eps = epsyy if tm else epsxx
        guess = np.max(eps)
        # need to solve for more than 2x modes since generally there will be two versions of each (H and V polarization) and we'll discard half
        hx,hy,neff = oct2py.octave.wgmodes(λ/1000, guess, nummodes, dx, dy, eps, boundary, nout=3)
    else:
        guess = np.max(epsyy) if tm else np.max(epsxx)
        hx,hy,neff = oct2py.octave.wgmodes(λ/1000, guess, nummodes, dx, dy, epsxx, epsyy, epszz, boundary, nout=3)
    if 3==len(hx.shape):    # more than one mode
        hxs,hys,neffs = hx[:,:,:],hy[:,:,:],[ni[0] for ni in neff]
        hx,hy,neff = hx[:,:,-1],hy[:,:,-1],neff[-1][0]
    else:                   # one mode
        hxs,hys,neffs = hx[:,:,np.newaxis],hy[:,:,np.newaxis],[neff]
    def power(ee):
        return (np.abs(ee)**2).mean()
    def istm(eex,eey):
        if verbose>1:
            print(' '*19+f"powery/powerx:{power(eey)/power(eex):g} {'HV'[power(eex)<power(eey)]}")
            # Wave2D(eex,xs=epsxx.xs,ys=epsxx.ys).plot(legendtext=f"{'HV'[power(eex)<power(eey)]}")

        return power(eex)<power(eey)
    def fields(neff,hhx,hhy):
        # note: hhz,eex,eey,eez are labeled according to wgmodes axes
        if isotropic:
            hhz,eex,eey,eez = oct2py.octave.postprocess(λ/1000, neff, hhx, hhy, dx, dy, eps, boundary, nout=4)
        else:
            hhz,eex,eey,eez = oct2py.octave.postprocess(λ/1000, neff, hhx, hhy, dx, dy, epsxx, epsyy, epszz, boundary, nout=4)
        return hhz,eex,eey,eez
    def poyntingvector(neff,hhx,hhy): # S = ExH = ExB/µ₀
        hhz,eex,eey,eez = fields(neff,hhx,hhy)
        def reshape(hh):
            return 0.25*( hh[1:,1:] + hh[1:,:-1] + hh[:-1,1:] + hh[:-1,:-1] )
        hx,hy,hz = reshape(hhx),reshape(hhy),reshape(hhz)
        # return eey*hz-hy*eez, eez*hx-hz*eex, eex*hy-hx*eey # Sx,Sy,Sz
        # return np.sqrt( (eey*hz-hy*eez)**2 + (eez*hx-hz*eex)**2 + (eex*hy-hx*eey)**2 ) # |S|
        return eex*hy-hx*eey # Sz
    def efield(neff,hhx,hhy):
        hhz,eex,eey,eez = fields(neff,hhx,hhy)
        # print(f" {neff:.5f} eex {100*power(eex):.4f} eey {100*power(eey):.4f}")
        if not tm==istm(eex,eey):
            if isotropic:
                return None # if the mode is the wrong polarization return None as a flag to filter them out
            if 1<verbose:
                print(f" {neff:.5f} eex {100*power(eex):.4f} eey {100*power(eey):.4f} {' x'[exact]}")
            # if not exact: print('neff',neff,'hhx,hhy',type(hhx),type(hhy))
            assert exact or np.isnan(neff), "modesolver mode has invalid polarization - fix not yet implemented, try method=isotropic"
            return None
        return eey if tm else eex
    ees = [efield(neffs[i], hxs[:,:,i], hys[:,:,i]) for i in range(len(neffs))]
    # hhs = [hxs[:,:,i] if power(hxs[:,:,i])>power(hys[:,:,i]) else hys[:,:,i] for i in range(len(neffs))]
    Ss = [poyntingvector(neffs[i], hxs[:,:,i], hys[:,:,i]) for i in range(len(neffs))]
    if (isotropic and verbose) or (exact and verbose):
        print(''.join([('VH' if tm else 'HV')[ee is None] for ee in ees]))
    if isotropic: # filter out modes of wrong polarization
        nes = [(neff,ee) for neff,ee in zip(neffs,ees) if ee is not None]
        assert nes, "correct polarization not found, try increasing nummodes"
        neffs,ees = zip(*nes)
        if verbose:
            print(f"   {len(neffs)} {'HV'[tm]} modes, {nummodes-len(neffs)} {'VH'[tm]} modes")
    elif exact:
        assert len(neffs)==nummodes
        nes = [(neff,ee) for neff,ee in zip(neffs,ees) if ee is not None]
        assert nes, "correct polarization not found, try increasing nummodes"
        neffs,ees = zip(*nes)
        if verbose:
            print(f"   {len(neffs)} {'HV'[tm]} modes, {nummodes-len(neffs)} {'VH'[tm]} modes")
    else:
        pass
    if len(neffs)<targetmode+1: # todo: flag to find all modes with dneff>0 (need targetindex?, do this in modesolver?)
        print(f'targeting {targetmode+1} modes, only found {len(neffs)}, now increasing nummodes from {nummodes} to {int(1.5*nummodes+1)}')
        return newoctavewgmodes(λ,epsxx,epsyy,epszz,int(1.5*nummodes+1),dx,dy,boundary,isotropic,exact,tm,verbose,targetmode)
    if ees[0].sum()<0: ees = [-ee if ee is not None else ee for ee in ees]
    # todo: normalize sign by choosing top left off-center to be positive
    neffs,ees = zip(*sorted(zip(neffs,ees),reverse=True)) # check that neffs are in order from biggest to smallest and if not reorder
    for i in range(len(neffs)-1):
        assert neffs[i]>=neffs[i+1] or str(neffs[i])=='nan' or str(neffs[i+1])=='nan', print(f'mode does not have smallest neff, i {i} i+1 {i+1} ni {neffs[i]} ni+1 {neffs[i+1]} neffs {neffs}')
    return neffs,ees,Ss

def fitgaussian2D(a,plotit=False,fitintensity=False,fitnorm=False):
    if a is None or np.isnan(a).any(): return Beamdata(np.nan,np.nan,np.nan,np.nan,np.nan)
    c = 2 if fitintensity else (0.5 if fitnorm else 1)
    def gaussian2D(p, amplitude, ωx, ωy, x0, y0, offset=0):
        x,y = p
        g = offset + amplitude*np.exp(-c*( ((x-x0)/ωx)**2 + ((y-y0)/ωy)**2 ))
        return g.ravel()
    a = a if abs(a.min())<abs(a.max()) else -a
    guess = (a.max(),1.1,1.2,*a.xymax())
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Covariance of the parameters could not be estimated")
            popt, pcov = scipy.optimize.curve_fit(gaussian2D, (a.xx, a.yy), a.ravel(), p0=guess)
        if plotit:
            psdev = np.sqrt(np.diag(pcov))
            Wave2D(gaussian2D((a.xx, a.yy),*popt).reshape(*a.shape),xs=a.xs,ys=a.ys).plot()
            print('best fit: '+' '.join(['%s:%#.4g±%#.4g'%(a,b,c) for a,b,c in zip('A ωx ωy x0 y0'.split(),popt,psdev)]))
        amplitude, ωx, ωy, x0, y0 = popt
        return Beamdata(amplitude, np.abs(ωx), np.abs(ωy), x0, y0)
    except:
        return Beamdata(*[np.nan]*5)


# λ in nm, all dimensions in µm
class Waveguide():
    @storecallargs
    def __init__(self,nx,ny=None,nz=None,λ=None,pol='v',sell=None):
        ny = ny if ny is not None else nx
        nz = nz if nz is not None else np.minimum(nx,ny)
        assert pol in ('h','v',None)
        self.nx,self.ny,self.nz,self.pol = nx,ny,nz,pol
        self.λ = λ
        self.sell = sell
        self.bounds,self.step = nx.bounds(),(nx.dx,nx.dy)
        assert np.allclose(nx.xs,ny.xs) and np.allclose(ny.xs,nz.xs)
        assert np.allclose(nx.ys,ny.ys) and np.allclose(ny.ys,nz.ys)
    @maplistargs
    def __call__(self,**kwargs):
        args = self.callargs.copy()
        args.update(kwargs)
        return type(self)(**args)
    def hindex(self):
        return self.nx
    def vindex(self):
        return self.ny
    def propindex(self):
        return self.nz
    @property
    def nn(self):
        return self.vindex() if 'v'==self.pol else self.hindex()
    def deltax(self):
        return self.nn.dx
    def deltay(self):
        return self.nn.dy
    def nsub(self):
        return index(self.λ,self.sell) if self.sell else self.nn.min()

    # def newmodesolve(self,method=None,mode=0,nummodes=None,boundary='0000',verbose=True,allmodes=False,**kwargs):
    #     if allmodes:
    #         return self(pol=None).solve(method=method,mode=mode,nummodes=nummodes,boundary=boundary,verbose=verbose,**kwargs)
    #     return self.solve(method=method,mode=mode,nummodes=nummodes,boundary=boundary,verbose=verbose,**kwargs)
    # def newmodesolve(self,method=None,mode=0,nummodes=None,boundary='0000',verbose=True,allmodes=False,**kwargs):
    #     # boundary NSEW, 'A' Hx/Hy antisymmetric/symmetric, 'S' Hx/Hy antisymmetric/symmetric, '0' Hx/Hy zero
    #     assert self.λ is not None
    #     assert self.pol in 'vh'
    #     boundary = boundary if boundary not in (None,'neumann') else 'SSSS' if 'v'==self.pol else 'AAAA' # todo: implement 'dirichlet' boundary conditions
    #     isotropic,exact = ('isotropic'==method),('exact'==method)
    #     neffs,ees,Ss = newoctavewgmodes(self.λ,self.hindex()**2,self.vindex()**2,self.propindex()**2,nummodes if nummodes is not None else 1+mode,
    #         self.deltax(),self.deltay(),boundary,isotropic=isotropic,exact=exact,tm=('v'==self.pol),verbose=verbose,targetmode=mode,**kwargs)
    #     ees = [Wave2D(ee,xs=self.nn.xs,ys=self.nn.ys) for ee in ees]
    #     # def reshape(hh): return 0.25*( hh[1:,1:] + hh[1:,:-1] + hh[:-1,1:] + hh[:-1,:-1] )
    #     # hhs = [Wave2D(reshape(hh),xs=self.nn.xs,ys=self.nn.ys) for hh in hhs]
    #     Ss = [Wave2D(S,xs=self.nn.xs,ys=self.nn.ys) for S in Ss]
    #     # mds = [Modedata0(self.λ,self.sell,n,self.nn,self.nsub(),ee,mode,self.pol,{}) for n,ee in zip(neffs,ees)]
    #     mds = [Modedata0(self.λ,self.sell,n,self.nn,self.nsub(),ee,S,mode,self.pol,{}) for n,ee,S in zip(neffs,ees,Ss)]
    #     # mds[mode].plot(); exit()
    #     return mds if allmodes else mds[mode]
    # def oldmodesolve(self,method=None,mode=0,nummodes=None,boundary='0000',verbose=True,allmodes=False,**kwargs):
    #     assert self.λ is not None
    #     assert self.pol in 'vh'
    #     # boundary = boundary if boundary is not None else ('00AA' if (sell.endswith('y') or xcut) else '00SS') if planar else '0000' # 'NSEW' = 'UDRL'
    #     isotropic,exact = ('isotropic'==method),('exact'==method)
    #     from modes import octavewgmodes
    #     from modes import Modedata0 as Oldmodedata
    #     if 0 and 'zhu'==method or 'zhuanisotropic'==method:
    #         import zhumodes
    #         εs = np.stack([ε.epsxx.np.T,ε.epsyy.np.T,ε.epszz.np.T], axis=2)
    #         tm = (not sell.endswith('y') and not d.xcut)
    #         neffs,eexs,eeys = zhumodes.zhumodesolve(lam=ε.λ/1000,n=sqrt(εs),x=ε.nn.xs,y=ε.nn.ys,nmodes=2*d.nummodes,
    #             method=None if 'zhuanisotropic'==d.method else 'tmisotropic' if tm else 'teisotropic')
    #         filtermodes = zhumodes.filtermodes(neffs,eexs,eeys,polarization='tm' if tm else 'te')
    #         neffs,ees = filtermodes
    #         def isreal(ee,tol=1e-9):
    #             return np.all(abs(ee.imag)<tol)
    #         if all([isreal(ee) for ee in ees]):
    #             ees = [-ee.real for ee in ees]
    #     neffs,ees = octavewgmodes(self.λ,self.hindex()**2,self.vindex()**2,self.propindex()**2,nummodes if nummodes is not None else 1+mode,
    #         self.deltax(),self.deltay(),boundary,isotropic=isotropic,exact=exact,tm=('v'==self.pol),verbose=verbose,targetmode=mode,**kwargs)
    #     ees = [Wave2D(ee,xs=self.nn.xs,ys=self.nn.ys) for ee in ees]
    #     # md = Modedata0(self.λ,self.sell,neffs,self.nn,self.nsub(),ees,mode,{})
    #     # md.pol = self.pol
    #     # mds = [Modedata0(self.λ,self.sell,n,self.nn,self.nsub(),ee,mode,self.pol,{}) for n,ee in zip(neffs,ees)]
    #     mds = [Oldmodedata(self.λ,self.sell,n,self.nn,self.nsub(),self.nn.max()-self.nsub(),[ee],mode,{}) for n,ee in zip(neffs,ees)]
    #     return mds if allmodes else mds[mode]
    #     # return mds[mode]
    #     if hasattr(self,'cca'):
    #         assert 0
    #         md.cca,md.ccr  = self.cca,self.ccr
    #     return md
    def zhusolve(self,method=None,mode=0,nummodes=None):
        # import zhumodes
        # εs = np.stack([ε.epsxx.np.T,ε.epsyy.np.T,ε.epszz.np.T], axis=2)
        # tm = (not sell.endswith('y') and not d.xcut)
        # neffs,eexs,eeys = zhumodes.zhumodesolve(lam=ε.λ/1000,n=sqrt(εs),x=ε.nn.xs,y=ε.nn.ys,nmodes=2*d.nummodes,
        #     method=None if 'zhuanisotropic'==d.method else 'tmisotropic' if tm else 'teisotropic')
        # filtermodes = zhumodes.filtermodes(neffs,eexs,eeys,polarization='tm' if tm else 'te')
        # neffs,ees = filtermodes
        # def isreal(ee,tol=1e-9):
        #     return np.all(abs(ee.imag)<tol)
        # if all([isreal(ee) for ee in ees]):
        #     ees = [-ee.real for ee in ees]
        import zhumodes
        # εs = np.stack([(self.hindex()**2).np.T,(self.vindex()**2).np.T,(self.propindex()**2).np.T], axis=2)
        ns = np.stack([self.hindex().np.T,self.vindex().np.T,self.propindex().np.T], axis=2)
        tm = ('v'==self.pol)
        neffs,eexs,eeys = zhumodes.zhumodesolve(lam=self.λ/1000,n=ns,x=self.nn.xs,y=self.nn.ys,nmodes=nummodes if nummodes is not None else 1+mode,method=method)
    def modesolve(self,method=None,mode=0,nummodes=None,boundary='0000',verbose=False,bothpolmodes=False,**kwargs):
        if bothpolmodes:
            return self(pol=None).solve(method=method,mode=mode,nummodes=nummodes,boundary=boundary,verbose=verbose,**kwargs)
        return self.solve(method=method,mode=mode,nummodes=nummodes,boundary=boundary,verbose=verbose,**kwargs)
        # return self.newmodesolve(method=method,mode=mode,nummodes=nummodes,boundary=boundary,verbose=verbose,bothpolmodes=bothpolmodes,**kwargs)
    def ms(self,*args,**kwargs):
        return self.modesolve(*args,**kwargs)
    def solve(self,solver='octave',method='supress',mode=0,nummodes=None,boundary=None,nsub=None,nguess=None,mretry=None,verbose=True):
        # for 'exact', we find all solutions and filter out wrong polarization (of which it may be all but the last few are)
        # for 'isotropic', need to solve for more than 2x modes since generally there will be two versions of each (H and V polarization) and we'll discard half
        # for 'supress', we modify the other axis using modifyeps() to be lower index if desired polarization is lower index than other axis
        λ,pol = self.λ,self.pol
        assert λ is not None and pol in ('h','v',None)
        method = method if method is not None else 'supress'
        assert method in ('exact','isotropic','supress')
        assert pol is not None or method in ('exact','isotropic') # if pol is None, return all modes unfiltered
        nummodes = nummodes if nummodes is not None else 1+mode
        mretry = mretry if mretry is not None else 3 if method in ('exact','isotropic') else 0 # retry multiplier, zero for no retry
        # an aside, showing mretry=3 is optimal: to find unknown mode number N, increasing the number of modes by a factor M each time takes 
        #   N(1+1/M+1/M^2+..)=Nm/(m-1) best case, N(M+1+1/M+1/M^2+..)=Nm²/(m-1) worst case, and N√M(M+1+1/M+1/M^2+..)=Nm^(3/2)/(m-1) average case.
        #   for average case the minimum price to pay occurs for m=3 with m^(3/2)/(m-1)=½√27=2.6.
        nx,ny,nz = 3*[self.vindex() if 'v'==pol else self.hindex()] if 'isotropic'==method else (self.hindex(),self.vindex(),self.propindex())
        epsx,epsy,epsz = nx**2,ny**2,nz**2
        dx,dy = self.deltax(),self.deltay()
        # print('mode',mode,'nummodes',nummodes)
        if 'supress'==method:
            def modifyeps(epsx,epsy,dn=0.1):
                # modesolver can't find the solution when desired polarization is lower index than other axis, so modify the other axis to be lower index. this is not done for 'exact' or 'isotropic'
                # dn=0.1 means index of axis opposite to the desired polarization will be set 0.1 lower
                def modified(eps,dn):
                    return (sqrt(eps)-dn)**2
                nx,ny = [np.max(sqrt(eps)) for eps in (epsx,epsy)]
                return (modified(epsx,nx+dn-ny),epsy) if 'v'==pol and ny<nx+dn else (epsx,modified(epsy,ny+dn-nx)) if 'v'!=pol and nx<ny+dn else (epsx,epsy)
            epsx,epsy = modifyeps(epsx,epsy)
            nx,ny = sqrt(epsx),sqrt(epsy)
        if 'octave'==solver:
            boundary = boundary if boundary is not None else '0000'
            b = ('SSSS' if 'v'==pol else 'AAAA') if 'n'==boundary[0] else ('AAAA' if 'v'==pol else 'SSSS') if 'd'==boundary[0] else boundary
            neffs,exs,eys,ezs,hxs,hys,hzs = octavesolver(λ,epsx,epsy,epsz,nummodes,dx,dy,b,method=method,nguess=nguess,verbose=verbose)
            # in octavesolver the E field is defined on cell centers but the H field is defined on cell edges
            # re-interpolate so all fields match the dimensions of the 2D index
            def reinterpolate(hh):
                return 0.25*( hh[1:,1:] + hh[1:,:-1] + hh[:-1,1:] + hh[:-1,:-1] )
            hxs,hys,hzs = ([reinterpolate(hh) for hh in hhs] for hhs in (hxs,hys,hzs))
        elif 'zhu'==solver:
            from zhumodes import zhumodesolve
            b = boundary if boundary is not None else 'neumann'
            # stack nx,ny,nz into 3D array, nnn.shape=(nx.shape[0],nx.shape[1],3)
            nnn = np.stack([nx.np,ny.np,nz.np], axis=2)
            neffs,exs,eys,ezs,hxs,hys,hzs = zhumodesolve(λ/1000,nnn,nx.xs,ny.ys,nummodes,nguess=nguess,method=method,boundary=b)
        else:
            assert 0
        exs,eys,ezs,hxs,hys,hzs = [[Wave2D(ee,xs=epsx.xs,ys=epsx.ys) for ee in ees] for ees in (exs,eys,ezs,hxs,hys,hzs)]
        nsub = nsub if nsub is not None else self.nsub()
        sell = self.sell if hasattr(self,'sell') else None
        wgargs = self.callargs.copy()
        # md = Modedata(λ,neffs,nx,ny,nz,exs,eys,ezs,hxs,hys,hzs,modenum=mode,pol=pol,nsub=nsub,sell=sell,wgargs=wgargs)
        md = Modedata(λ=λ,neffs=neffs,nx=nx,ny=ny,nz=nz,
                      Exs=exs,Eys=eys,Ezs=ezs,Hxs=hxs,Hys=hys,Hzs=hzs,
                      modenum=mode,pol=pol,nsub=nsub,sell=sell,wgargs=wgargs)
        # return md.filterpolarization(pol,verbose) if pol in ('h','v') else md
        if pol not in ('h','v'):
            return md
        mdf = md.filterpolarization(pol,verbose)
        if not isinstance(mode,int):
            raise NotImplementedError('string or tuple mode not yet implemented')
            # todo
            # Modedata modenum update to accept string or tuple
            # handling modeid: if mode is string or tuple, check if in md.modeids(), if not, retry
            # Modedata handles modecount()<=modenum?
            # if one or more modes found, estimate a better nummodes to try next?
        numretry = int(np.ceil(mretry*nummodes+1e-9)) if 0<mretry else 0 # print('mretry',mretry,'numretry',numretry)
        if 0==numretry or mode<len(mdf.neffs): # print('guidedmodecount',mdf.guidedmodecount(),mdf.dneffs)
            return mdf
        if verbose: print(f"found {len(mdf.neffs)} modes only, increasing nummodes from {nummodes} to {numretry}")
        # print('retrying:','numretry',numretry,'mode',mode,'modecount',len(mdf.neffs),'Δn',mdf.dneffs)
        assert nummodes<numretry, f"nummodes must be less than numretry, nummodes:{nummodes} numretry:{numretry}"
        return self.solve(solver=solver,method=method,mode=mode,nummodes=numretry,boundary=boundary,nsub=nsub,nguess=nguess,mretry=mretry,verbose=verbose)

    def singleysplittercouplingphase(self,sx,sy,width,splitradius=0,leftrightsymmetric=True,effectivelength=False,plot=False):
        # width = input width, splitradius = radius of the splitter crotch
        # coupler width = 2x input width
        from sellmeier import sbendroc,sbend # print('sx',sx,'sy',sy,'roc in µm',sbendroc(L=sx,w=sy))
        assert leftrightsymmetric
        splits = wrange(width+2*splitradius,
                        width+2*splitradius+8,1,aslist=1)        
        xs = np.linspace(0,sx,101)/1000
        ys = (2 if leftrightsymmetric else 1)*sbend(1000*np.abs(xs),sx,sy) + width
        self(split=np.array(splits)).diffusionprecompute()
        Lcs = [self(w=width,split=split).modesolve(mode=1).couplinglength(atol=1e-4,plot=0) for split in splits] # 
        if splitradius:
            L0 = self(w=2*width,            split=0).modesolve(mode=1).couplinglength(atol=1e-4,plot=0)
            L1 = self(w=2*width+splitradius,split=0).modesolve(mode=1).couplinglength(atol=1e-4,plot=0)
            Lcs = [L0,L1] + Lcs
            splits = [width,splits[0]] + list(splits)
        Lcwave = Wave(Lcs,splits).removenans() # print('Lcwave',Lcwave)
        if plot: Wave().plots(Lcwave,m='o',log=1,grid=1,x='split (µm)',y='coupling length (µm)',xlim=(splits[0],splits[-1]))
        def µ(y):
            return np.pi/2 * 1/Lcwave(y,extrapolate='log')
        ϕ = np.trapz([µ(y) for y in ys],x=xs) # integrated phase over the propagation distance
        return ϕ if not effectivelength else ϕ*2/pi*Lcs[0]

    def singlesbendcouplingphase(self,sx,sy,splits,leftrightsymmetric=False,effectivelength=False,plot=False):
        from sellmeier import sbendroc,sbend
        # print('sx',sx,'sy',sy,'roc in µm',sbendroc(L=sx,w=sy))
        xs = np.linspace(0,sx,101)/1000
        ys = (2 if leftrightsymmetric else 1)*sbend(1000*np.abs(xs),sx,sy) + splits[0]
        self(split=splits).diffusionprecompute()
        Lcs = [self(split=split).modesolve(mode=1).couplinglength(atol=1e-4,plot=0) for split in splits] # print('Lcs',Lcs)
        Lcwave = Wave(Lcs,splits).removenans() # print('Lcwave',Lcwave)
        if plot: Wave().plots(Lcwave,m='o',log=1,grid=1,x='split (µm)',y='coupling length (µm)',xlim=(splits[0],splits[-1]))
        def µ(y):
            return np.pi/2 * 1/Lcwave(y,extrapolate='log')
        ϕ = np.trapz([µ(y) for y in ys],x=xs) # integrated phase over the propagation distance
        return ϕ if not effectivelength else ϕ*2/pi*Lcs[0]

    def singletapercouplingphase(self,tx,width0,width1,style='cos',effectivelength=False,plot=False): # tx in µm, width0,width1 in µm
        assert style in ('linear','cos')
        def f(x):
            return width0+(width1-width0)*x/tx if 'linear'==style else width0+(width1-width0)*0.5*(1-np.cos(np.pi*x/tx))
        xs = np.linspace(0,tx,101)/1000 # tx in µm, xs in mm
        ys = f(xs) # print('ys',ys)
        widths = wrange(width0,width1,1)
        self(w=widths,split=0).diffusionprecompute()
        Lcs = [self(w=width,split=0).modesolve(mode=1).couplinglength(atol=1e-4,plot=0) for width in widths]; # print('Lcs',Lcs)
        Lcwave = Wave(Lcs,widths).removenans(); # print('Lcwave',Lcwave)
        if plot: Wave().plots(Lcwave,m='o',log=1,grid=1,x='width (µm)',y='coupling length (µm)',xlim=(width0,width1))
        def µ(y):
            return np.pi/2 * 1/Lcwave(y,extrapolate='log')
        ϕ = np.trapz([µ(y) for y in ys],x=xs) # integrated phase over the propagation distance
        return ϕ if not effectivelength else ϕ*2/np.pi*Lcs[0]

    def constantgapcouplingphase(self,tx,width0,width1,gap,style='cos',effectivelength=False,plot=False): # tx in µm, width0,width1 in µm, gap = edge-to-edge gap in µm
        assert style in ('linear','cos')
        def f(x):
            return width0+(width1-width0)*1000*x/tx if 'linear'==style else width0+(width1-width0)*0.5*(1-np.cos(np.pi*1000*x/tx))
        xs = np.linspace(0,tx,101)/1000 # tx in µm, xs in mm
        ys = f(xs) # print('ys',ys)
        widths = wrange(width0,width1,1)
        self(w=widths,split=gap+widths).diffusionprecompute()
        Lcs = [self(w=width,split=gap+width).modesolve(mode=1).couplinglength(atol=1e-4,plot=0) for width in widths]; # print('Lcs',Lcs)
        Lcwave = Wave(Lcs,widths).removenans(); # print('Lcwave',Lcwave)
        if plot: Wave().plots(Lcwave,m='o',log=1,grid=1,x='width (µm)',y='coupling length (µm)',xlim=(width0,width1))
        def µ(y):
            return np.pi/2 * 1/Lcwave(y,extrapolate='log')
        ϕ = np.trapz([µ(y) for y in ys],x=xs) # integrated phase over the propagation distance
        return ϕ if not effectivelength else ϕ*2/np.pi*Lcs[0]

    def qpm(self,λ1,λ2=None,Type='vvv',dmask=None,method=None,modes=(0,0,0),nummodes=(None,None,None),boundary='0000',**kwargs):
        λ1,λ2,λ3 = qpmwavelengths(λ1,λ2)
        # mds = [md.modesolve() for md in self(λ=list(λs),pol=list(Type))]
        mds = [self(λ=λ,pol=p).modesolve(method=method,mode=mode,nummodes=nm,boundary=boundary) for λ,p,mode,nm in zip((λ1,λ2,λ3),Type,modes,nummodes)]
        if dmask is None and hasattr(self,'dmask'):
            kwargs['dmask'] = self.dmask()
        if dmask is not None:
            kwargs['dmask'] = dmask
        return Qpmdata(*mds,kwargs)
    def λ1qpm(self,Λ,λ1,λ2=None,Δλ=100,λtol=1,verbose=False,Type='vvv',dmask=None,method=None,modes=(0,0,0),nummodes=(None,None,None),boundary='0000',**kwargs):
        from wavedata import finvert
        args = dict(Type=Type,dmask=dmask,method=method,modes=modes,nummodes=nummodes,boundary=boundary,**kwargs)
        def func(λ):
            return 1/self.qpm(λ1=λ,λ2=λ2 if λ2 is not None else λ,**args).Λ - 1/Λ
        return finvert(func,x0=λ1-Δλ,x1=λ1+Δλ,xtol=λtol,verbose=verbose)
    def λ2qpm(self,Λ,λ1,λ2,Δλ=100,λtol=1,verbose=False,Type='vvv',dmask=None,method=None,modes=(0,0,0),nummodes=(None,None,None),boundary='0000',**kwargs):
        from wavedata import finvert
        args = dict(λ1=λ1,Type=Type,dmask=dmask,method=method,modes=modes,nummodes=nummodes,boundary=boundary,**kwargs)
        def func(λ):
            return 1/self.qpm(λ2=λ,**args).Λ - 1/Λ
        return finvert(func,x0=λ2-Δλ,x1=λ2+Δλ,xtol=λtol,verbose=verbose)
    def λshgqpm(self,Λ,λ1,λ2=None,Δλ=100,λtol=1,verbose=False,Type='vvv',dmask=None,method=None,modes=(0,0,0),nummodes=(None,None,None),boundary='0000',**kwargs):
        assert λ1==λ2 or λ2 is None
        return self.λ1qpm(Λ,λ1,λ2=None,Δλ=Δλ,λtol=λtol,verbose=verbose,Type=Type,dmask=dmask,method=method,modes=modes,nummodes=nummodes,boundary=boundary,**kwargs)
    def λqpm(self,Λ,λ1,λ2=None,Δλ=100,λtol=1,verbose=False,Type='vvv',dmask=None,method=None,modes=(0,0,0),nummodes=(None,None,None),boundary='0000',**kwargs):
        return self.λ1qpm(Λ,λ1,λ2=λ2,Δλ=Δλ,λtol=λtol,verbose=verbose,Type=Type,dmask=dmask,method=method,modes=modes,nummodes=nummodes,boundary=boundary,**kwargs)

    # def idstr(self,short=False):
    #     λ1,λ2,w,d,a,r,aa = self.λ1,self.λ2,self.w,self.d,self.a,self.r,self.aa
    #     s = f"{λ1:g}+{λ2:g}" if short else f"{λ1:g}+{λ2:g} {w:g}w {d:g}sa {a:g}a"
    #     s += f" {r:g}r"*bool(r)+f" {aa:g}a2"*bool(aa)
    #     s += f" {self.atemp:g}°a"*bool(320!=self.atemp)+f" {self.rtemp:g}°r"*bool(300!=self.rtemp)
    #     return s
    def Λvswidth(self,ws=None,temps=None,plot=False,save=None,**plotargs):
        from wavedata import Wave
        save = save if save is not None else f"Λ vs width" #, {self.idstr()}"
        ws = ws if ws is not None else np.linspace(6,12,7)
        us = [Wave([self(w=w).Λ(temp=temp) for w in ws],ws,f"{temp}°C" if temps is not None else "").setplot(c=i) for i,temp in enumerate(temps if temps is not None else [None])]
        u0s = [u.quadminloc(aswave=1).setplot(m='o',l=' ',c=i).rename(f"Λ={u.quadmin():.3f}µm") for i,u in enumerate(us)]
        if plot or plotargs:
            Wave.plots(*us,*u0s,x='width (µm)',y='Λ (µm)',legendtext=f"noncrit width {us[0].quadminloc():.1f}µm",corner='upper right',save=save,seed=int(self.λ1+self.λ2),showseed=1,**plotargs)
            return f"figs/{save}.png"
        return us if temps is not None else us[0].setplot(c=None)

    def rotate(self):
        nx = self.nx.transpose(mirrory=True)
        ny = self.ny.transpose(mirrory=True)
        nz = self.nz.transpose(mirrory=True)
        limits = self.limits[2:]+self.limits[:2]
        return Dielectric(nx=nx,ny=ny,nz=nz,
            nsub=self.nsub(),dns=self.dns,λ=self.λ,res=self.res,limits=limits,pol=self.pol)
    def plot(self,**kwargs):
        # self.nn.plot(vmin=self.nsub()-0.001,vmax=self.nsub()+self.dns,**kwargs)
        self.nn.plot(**kwargs)
    def plotx(self,**kwargs):
        Wave().plots(*[self.nn.aty(y) for y in [self.nn.ymin,-self.nn.dy,self.nn.ymax]],ylim=(self.nsub()-0.001,self.nsub()+self.dns+0.001),**kwargs)
    def ploty(self,**kwargs):
        Wave().plots(*[self.nn.atx(x) for x in [0,self.nn.xmax]],ylim=(self.nsub()-0.001,self.nsub()+self.dns+0.001),**kwargs)
    def __str__(self):
        return f"λ:{self.λ}, n:{self.nsub():.4f}, step:{self.step}, bounds:{self.bounds}, nn.shape:{self.nn.shape}"
class Boxwaveguide(Waveguide):
    @storecallargs
    def __init__(self,w,h,n,n0=1,λ=None,pol='v',sell=None,bounds=None,step=(0.2,0.2)):
        x0,x1,y0,y1 = bounds = bounds if bounds is not None else (-w/2-1,w/2+1,-h/2-1,h/2+1)
        stepx,stepy = step if hasattr(step,'__len__') else (step,step)
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        nn = n0 + (n-n0)*nn.centeredrectangle(w,h)
        super().__init__(nn,nn,nn,λ,pol,sell)
class Planewaveguide(Boxwaveguide):
    @storecallargs
    def __init__(self,λ=None,pol='v',n=None,sell=None,bounds=None,step=0.2):
        self.sell = sell if sell is not None else 'air'
        x0,x1,y0,y1 = bounds = bounds if hasattr(bounds,'__len__') else np.array((-1,1,-1,1))*bounds if bounds is not None else np.array((-1,1,-1,1))
        n = index(λ,self.sell) if n is None else n
        w,h = 0.5*(x1-x0),0.5*(y1-y0)
        super().__init__(w,h,n+1e-9,n0=n,λ=λ,pol=pol,sell=sell,bounds=bounds,step=step)
class Stepfiberwaveguide(Waveguide):
    @storecallargs
    def __init__(self,r,ncore=None,nclad=None,λ=None,pol='v',sell=None,bounds=None,step=0.2):
        self.r = r
        self.nclad = nclad if nclad is not None else index(λ,'sio2') if λ is not None else 1.45
        self.ncore = ncore if ncore is not None else self.nclad+0.005
        x0,x1,y0,y1 = bounds if hasattr(bounds,'__len__') else np.array((-1,1,-1,1))*bounds if bounds is not None else np.array((-1,1,-1,1))*(10+np.ceil(r))
        stepx,stepy = step if hasattr(step,'__len__') else (step,step)
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        nn = self.nclad + (self.ncore-self.nclad)*nn.circle(0,0,r)
        super().__init__(nn,nn,nn,λ,pol,sell)
    def fibermode(self,l=0,x0=0,y0=0,res=None,limits=None,θx=0,θy=0): # λ in nm, a in µm
        λ,a,n,dn = self.λ,self.r,self.nclad,self.ncore-self.nclad
        xmin,xmax,ymin,ymax = limits = limits if limits is not None else self.bounds
        res = res if res is not None else self.step
        args = locals()
        V = (2*np.pi*1000/λ)*a*np.sqrt((n+dn)**2-n**2)
        ωapprox = a*(0.65 + 1.619/V**1.5 + 2.879/V**6)
        from scipy.special import jv,kv
        # wx = Wave(index=np.linspace(0,5,51)); Wave().plots(*[jv(n,wx) for n in [0,1,2,3]] + [kv(n,wx) for n in [0,1,2,3]],ylim=(-1,2),groups=4)
        # https://www.rp-photonics.com/step_index_fibers.html # gloge1971 - Weakly Guiding Fibers.pdf
        # Ch 4, Step-index fibers (STU).pdf # https://www.hft.tu-berlin.de/fileadmin/fg154/ONT/Skript/ENG-Ver/STU_06_05.pdf
        def fl(b,l):
            u,v = V*np.sqrt(1-b),V*np.sqrt(b)
            return u*jv(l+1,u)/jv(l,u) - v*kv(l+1,v)/kv(l,v)
        def einside(r,b,l):
            return jv(l,V*np.sqrt(1-b)*r/a) * (np.abs(r)<=a)
        def eoutside(r,b,l):
            x = kv(l,V*np.sqrt(b)*np.abs(r)/a) * jv(l,V*np.sqrt(1-b)*np.sign(r)) / kv(l,V*np.sqrt(b)); #print('type(x)',type(x))
            x = np.nan_to_num(np.array(x))
            return x * (a<=np.abs(r))
        def eapprox(r):
            ω = a*(0.65 + 1.619/V**1.5 + 2.879/V**6)
            return np.exp(-r**2/ω**2)
        # wx = Wave(index=np.linspace(1e-9,1,1001)); Wave().plots(*[fl(wx,n) for n in [0,1,2]],ylim=(-100,100))
        xs = np.linspace(1e-9,1,1001)
        zs = np.diff(np.sign(fl(xs,l))) # diff(sign(w)) will be non-zero at cross-overs and will have the sign of the crossing: # https://stackoverflow.com/a/25091643
        zerocrossings = np.where(-2==zs)[0] # eg. zs = [0 0 0 0 0 0 0 0 2 0 0 0 -2 0 0 0]
        # args['zerocrossings'] = zerocrossings # print('approximate zeros at',[xs[z] for z in zerocrossings]) # + to - are the crossings we want, - to + are -∞ to +∞
        # wx = Wave(index=np.linspace(1e-9,xs[zerocrossings[3]],101)); Wave().plots(fl(wx,0),0*wx,pause=0)
        # if plotzerocrossings: wx = Wave(index=xs); Wave().plots(fl(wx,0),0*wx,ylim=(-100,100),pause=0)
        if 0==len(zerocrossings): assert 0,'no modes found, try l=0'
        z0 = zerocrossings[-1] # last crossing is the fundamental mode, others are higher order modes
        b0 = scipy.optimize.brentq(lambda b:fl(b,l), xs[z0], xs[z0+1])
        neff = b0*dn+n
        # if plot: wx = Wave(index=np.linspace(-2*a,2*a,101)); Wave().plots(einside(wx,b0,l),eoutside(wx,b0,l),eapprox(wx),pause=0)#,ylim=(None,1e-2))
        # make 2D data
        # xx,yy = Wave2D(xs=np.arange(xmin,xmax+res/2,res),ys=np.arange(ymin,ymax+res/2,res),returngrid=1)
        resx,resy = res if hasattr(res,'__len__') else [res,res]
        xx,yy = Wave2D(xs=np.arange(xmin,xmax+resx/2,resx),ys=np.arange(ymin,ymax+resy/2,resy)).grid()
        rr,θθ = np.sqrt((xx-x0)**2+(yy-y0)**2),np.arctan2(yy-y0,xx-x0)
        def step(x,v0=0.5,dx=0): # returns 0 for x<0, 1 for 0<x, v0 is value at x==0
            if dx: return scipy.special.erfc(-x/dx)/2
            return np.heaviside(x,v0)
        nn = n + dn*step(a-rr,dx=min(resx,resy)) # nn.plot()
        ee = (step(a-rr)*einside(rr,b0,l) + (1-step(a-rr))*eoutside(rr,b0,l)) * np.cos(l*(θθ-np.pi/2)) # ee.plot(legendtext='fibermode ee',pause=1)
        if θx or θy:
            ee = ee * np.exp(1j*2000*pi*θx*xx/λ) * np.exp(1j*2000*pi*θy*yy/λ)
        # Modedata0(self.λ,self.sell,n,self.nn,self.nsub(),ee,S,mode,self.pol,{})
        # md = Modedata0(λ,'sio2',neff,nn,n,ee,None,l,None,args) # md.ex.plot()
        md = Modedata(λ,[neff],nn,nn,nn,[0*ee],[ee],[0*ee],[nan*ee],[nan*ee],[nan*ee],modenum=l,pol='v',nsub=n,sell='sio2',wgargs=args)
        return md
class Corningpmfiberwaveguide(Stepfiberwaveguide):
    fibers = [460,630,780,980,1550,2000]
    pmfibercutoff = {460:410, 630:570, 780:710, 980:920, 1300:1210, 1550:1380, 2000:2000}
    pmfibercorediameter = {460:3.0, 630:3.5, 780:4.5, 980:5.5, 1300:8.0, 1550:8.5, 2000:7.0}
    pmcoreindex = {460:1.4697, 630:1.4617, 780:1.4587,980:1.4555,1550:1.449,2000:1.462}  # from Thorlabs tech support 11/13/18
    pmcladdingindex = {460:1.4648, 630:1.4571, 780:1.4537, 980:1.4507, 1550:1.444, 2000:1.449} # Δn = 0.005
    pmindexwavelength = {460:460, 630:630, 780:780, 980:980, 1550:1550, 2000:2000}
    pmmfd = {460:3.3, 630:4.5, 780:5.3, 980:6.6, 1300:9.3, 1400:9.8, 1550:10.1, 2000:8.0}
    pmmfdwavelength = {460:515, 630:630, 780:850, 980:980, 1300:1300, 1400:1450, 1550:1550, 2000:1950}
    @storecallargs
    def __init__(self,fiber,λ=None,pol='v',sell=None,bounds=None,step=0.2):
        r = self.__class__.pmfibercorediameter[fiber]/2
        ncore = self.__class__.pmcoreindex[fiber]
        nclad = self.__class__.pmcladdingindex[fiber]
        super().__init__(r,ncore,nclad,λ,pol,sell,bounds=bounds,step=step)

class Anisowaveguide(Waveguide): # anisotropic waveguide with separate index for each axis
    def __init__(self,crystal='ln',pol='v',cut='zyx',λ=None,bounds=None,deadbounds=None,step=(0.2,0.2)):
        # crystal must have sellmeier defined for each axis 
        self.bounds = bounds
        self.step = step if hasattr(step,'__len__') else (step,step)
        assert 3==len(cut)
        pol = 'vh'[cut.index(pol)] if pol and pol in cut else pol # e.g. converts Type 'yzy' to 'hvh'
        assert pol in ('h','v',None)
        self.pol,self.cut = pol,cut # v,h,prop
        self.crystal = crystal
        self.sell = crystal + cut[('h'==pol)]
        self.λ = λ
        self.deadbounds = deadbounds
    def vindex(self):
        return self.indexfunc(self.λ,self.cut[0])
    def hindex(self):
        return self.indexfunc(self.λ,self.cut[1])
    def propindex(self):
        return self.indexfunc(self.λ,self.cut[2])
    def nsub(self):
        return index(self.λ,self.sell)
    def indexfunc(self,λ,axis):
        pass
class Ridgewaveguide(Anisowaveguide):
    @storecallargs
    def __init__(self,w=10,h=10,etch=None,bf=1,split=0,crystal='mgln',sellbase='sio2',sellcover='air',pol='v',cut='zyx',λ=None,bounds=None,step=0.2):
        etch = etch if etch is not None else h
        self.w,self.h,self.etch,self.bf,self.split = w,h,etch,bf,split
        self.sellbase,self.sellcover = sellbase,sellcover
        bounds = bounds if bounds is not None else 5
        bounds = bounds if hasattr(bounds,'__len__') else (-(0.5*w+0.5*split+bounds),(0.5*w+0.5*split+bounds),-(h+bounds),bounds)
        super().__init__(crystal=crystal,pol=pol,cut=cut,λ=λ,bounds=bounds,step=step)
    def indexfunc(self,λ,axis):
        nridge,nbase,nair = [index(λ,s) for s in (self.crystal+axis,self.sellbase,self.sellcover)]
        w,h,etch,bf,s = self.w,self.h,self.etch,self.bf,self.split
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        n0 = nn.yslabs(ys=[-h,-etch],ns=[nbase,nridge,nair])
        n1 = bf*nn.yslabs(ys=[-h,0],ns=[nbase,nridge,nair]) + (1-bf)*n0
        xs = [-0.5*(w+s),0.5*(w+s)] if s<w else [0.5*(-s-w),0.5*(-s+w),0.5*(+s-w),0.5*(+s+w)]
        ns = [n0,n1,n0] if s<w else [n0,n1,n0,n1,n0]
        return nn.xslabs(xs=xs,ns=ns)
    def nsub(self):
        return index(self.λ,self.sellbase)
class Xcutridgewaveguide(Ridgewaveguide):
    @storecallargs
    def __init__(self,w=10,h=10,etch=None,bf=1,split=0,crystal='mgln',sellbase='sio2',sellcover='air',pol='h',cut='xzy',λ=None,bounds=None,step=0.2):
        super().__init__(w=w,h=h,etch=etch,bf=bf,split=split,crystal=crystal,sellbase=sellbase,sellcover=sellcover,pol=pol,cut=cut,λ=λ,bounds=bounds,step=step)
    def qpm(self,λ1,λ2=None,Type='hhh',dmask=None,method=None,modes=(0,0,0),nummodes=(None,None,None),boundary='0000',**kwargs):
        return super().qpm(λ1,λ2,Type,dmask,method,modes,nummodes,boundary,**kwargs)
class Ridgewaveguidecarrier(Anisowaveguide):
    @storecallargs
    def __init__(self,w=10,h=10,b=0.5,etch=None,bf=1,split=0,crystal='mgln',sellbase='sio2',sellcover='air',sellcarrier='mgln',pol='v',cut='zyx',λ=None,bounds=None,step=0.2):
        etch = etch if etch is not None else h
        self.w,self.h,self.b,self.etch,self.bf,self.split = w,h,b,etch,bf,split
        self.sellbase,self.sellcover,self.sellcarrier = sellbase,sellcover,sellcarrier
        bounds = bounds if bounds is not None else 5
        bounds = bounds if hasattr(bounds,'__len__') else (-(0.5*w+0.5*split+bounds),(0.5*w+0.5*split+bounds),-(h+b+bounds),bounds)
        super().__init__(crystal=crystal,pol=pol,cut=cut,λ=λ,bounds=bounds,step=step)
    def indexfunc(self,λ,axis):
        nridge,nbase,nair,ncarrier = [index(λ,s) for s in (self.crystal+axis,self.sellbase,self.sellcover,self.sellcarrier)]
        w,h,b,etch,bf,s = self.w,self.h,self.b,self.etch,self.bf,self.split
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        ys = [-h-b,-h,-etch] if etch<h else [-h-b,-etch] if etch<h+b else [-etch]
        ns = [ncarrier,nbase,nridge,nair] if etch<h else [ncarrier,nbase,nair] if etch<h+b else [ncarrier,nair]
        n0 = nn.yslabs(ys=ys,ns=ns)
        n1 = bf*nn.yslabs(ys=[-h-b,-h,0],ns=[ncarrier,nbase,nridge,nair]) + (1-bf)*n0

        xs = [-0.5*(w+s),0.5*(w+s)] if s<w else [0.5*(-s-w),0.5*(-s+w),0.5*(+s-w),0.5*(+s+w)]
        ns = [n0,n1,n0] if s<w else [n0,n1,n0,n1,n0]
        return nn.xslabs(xs=xs,ns=ns)
    def nsub(self):
        return index(self.λ,self.sellbase)

class Trapezoidalridgewaveguide(Ridgewaveguide):
    @storecallargs
    def __init__(self,w=10,h=10,etch=None,bf=1,split=0,crystal='mgln',sellbase='sio2',sellcover='air',pol='v',cut='zyx',λ=None,bounds=None,step=0.2,roc=0,trapezoidangle=0):
        assert -np.pi/2<trapezoidangle<np.pi/2
        self.roc,self.trapezoidangle = roc,trapezoidangle # sidewall angle from vertical in radians, etch width is the width at the top of the trapezoid
        super().__init__(w=w,h=h,etch=etch,bf=bf,split=split,crystal=crystal,sellbase=sellbase,sellcover=sellcover,pol=pol,cut=cut,λ=λ,bounds=bounds,step=step)
    def indexfunc(self,λ,axis):
        nridge,nbase,nair = [index(λ,s) for s in (self.crystal+axis,self.sellbase,self.sellcover)]
        w,h,etch,bf,s,roc = self.w,self.h,self.etch,self.bf,self.split,self.roc
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        def wvsd(d): # ridge width vs depth (depth = |depth| = -y)
            return w+2*d*np.tan(self.trapezoidangle)
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        n0 = nair + bf*(nridge-nair)*tophat(nn.xx,-0.5*wvsd(-nn.yy),0.5*wvsd(-nn.yy),nn.dx)
        if s:
            n0 = nair + bf*(nridge-nair)*np.maximum(tophat(nn.xx,0.5*(-s-wvsd(-nn.yy)),0.5*(-s+wvsd(-nn.yy)),nn.dx),
                                                    tophat(nn.xx,0.5*(+s-wvsd(-nn.yy)),0.5*(+s+wvsd(-nn.yy)),nn.dx))
        n1 = nn.yslabs(ys=[-h,-etch,0],ns=[nbase,nridge,n0,nair])
        # if roc:assert 0<1-0.001*n1.xs.min()/roc;return n1 * (1+0.001*n1.xx/roc)
        return n1 * np.exp(0.001*n1.xx/roc) if roc else n1
class Xcuttrapezoidalridgewaveguide(Trapezoidalridgewaveguide):
    @storecallargs
    def __init__(self,w=10,h=10,etch=None,bf=1,split=0,crystal='mgln',sellbase='sio2',sellcover='air',pol='h',cut='xzy',λ=None,bounds=None,step=0.2,roc=0,trapezoidangle=0):
        super().__init__(w=w,h=h,etch=etch,bf=bf,split=split,crystal=crystal,sellbase=sellbase,sellcover=sellcover,pol=pol,cut=cut,λ=λ,bounds=bounds,step=step,roc=roc,trapezoidangle=trapezoidangle)
    def qpm(self,λ1,λ2=None,Type='hhh',dmask=None,method=None,modes=(0,0,0),nummodes=(None,None,None),boundary='0000',**kwargs):
        return super().qpm(λ1,λ2,Type,dmask,method,modes,nummodes,boundary,**kwargs)

class Sinwaveguide(Anisowaveguide):
    @storecallargs
    def __init__(self,w=2,wfn=None,d=0.22,dfn=None,ygap=0.05,bf=1,bffn=None,crystal='sin',sellbase='sio2',sellcover='air',pol='v',cut='zyx',λ=None,bounds=None,step=(0.2,0.05),split=0):
        # fn,sn = first nitride layer, second nitride layer # bf = bragg fraction
        wfn,dfn,bffn = wfn if wfn is not None else w, dfn if dfn is not None else d, bffn if bffn is not None else bf
        self.w,self.wfn,self.d,self.dfn,self.ygap,self.bf,self.bffn,self.split = w,wfn,d,dfn,ygap,bf,bffn,split
        self.sellbase,self.sellcover = sellbase,sellcover
        margin,x0 = 5, 0.5*split/2 + 0.5*max(w,wfn)
        bounds = bounds if bounds is not None else (-x0-margin,x0+margin,-1-margin,margin)
        super().__init__(crystal=crystal,pol=pol,cut=cut,λ=λ,bounds=bounds,deadbounds=None,step=step)
    def indexfunc(self,λ,axis):
        nridge,nbase,nair = [index(λ,s) for s in (self.crystal+axis,self.sellbase,self.sellcover)]
        w,wfn,d,dfn,ygap,bf,bffn,s = self.w,self.wfn,self.d,self.dfn,self.ygap,self.bf,self.bffn,self.split
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        # if 0==split: 
        #     nnsn = (1-bf)*nbase + bf*nn.xslabs(xs=[-0.5*w,0.5*w],ns=[nbase,nridge,nbase])
        #     nnfn = (1-bffn)*nbase + bffn*nn.xslabs(xs=[-0.5*wfn,0.5*wfn],ns=[nbase,nridge,nbase])
        #     return nn.yslabs(ys=[-dfn-ygap-d,-ygap-d,-d,0],ns=[nbase,nnfn,nbase,nnsn,nair])
        # nnsn = (1-bf)*nbase + bf*nn.xslabs(xs=[0.5*(-split-w),0.5*(-split+w),0.5*(+split-w),0.5*(+split+w)],ns=[nbase,nridge,nbase,nridge,nbase])
        # nnfn = (1-bffn)*nbase + bffn*nn.xslabs(xs=[0.5*(-split-wfn),0.5*(-split+wfn),0.5*(+split-wfn),0.5*(+split+wfn)],ns=[nbase,nridge,nbase,nridge,nbase])
        # return nn.yslabs(ys=[-dfn-ygap-d,-ygap-d,-d,0],ns=[nbase,nnfn,nbase,nnsn,nair])
        xsn = [-0.5*(w+s),0.5*(w+s)] if s<w else [0.5*(-s-w),0.5*(-s+w),0.5*(+s-w),0.5*(+s+w)]
        nsn = [nbase,nridge,nbase] if s<w else [nbase,nridge,nbase,nridge,nbase]
        nnsn = (1-bf)*nbase + bf*nn.xslabs(xs=xsn,ns=nsn)
        xfn = [-0.5*(wfn+s),0.5*(wfn+s)] if s<wfn else [0.5*(-s-wfn),0.5*(-s+wfn),0.5*(+s-wfn),0.5*(+s+wfn)]
        nfn = [nbase,nridge,nbase] if s<wfn else [nbase,nridge,nbase,nridge,nbase]
        nnfn = (1-bffn)*nbase + bffn*nn.xslabs(xs=xfn,ns=nfn)
        return nn.yslabs(ys=[-dfn-ygap-d,-ygap-d,-d,0],ns=[nbase,nnfn,nbase,nnsn,nair])
    def nsub(self):
        return index(self.λ,self.sellbase)
    def meshplot(self):
        from wavedata import Mesh2Ds
        names = 'air SiO₂ SiN'
        m = Mesh2D(names,xs=[],ys=[])
        self.plot()
def ktp2d(y,z,λ,width=3,depth=5,conc=1,pol='z'):
    # 2D index profile for z-cut KTP waveguide
    # y = horizontal position in µm, z = vertical position in µm (z<0 is in the KTP, z>0 is air)
    # λ = wavelength in nm, width = width of KTP waveguide in µm, depth = FWHM depth of KTP waveguide in µm, conc = concentration of KTP (0<conc<=1)
    # pol = polarization (z or y)
    # bulk index ref: KatoTakaoka2002 - Sellmeier and thermo-optic dispersion formulas for KTP
    # surface index ref: Callahan2014 - Fiber-coupled balanced optical cross-correlator using PPKTP waveguides (has typos)
    import numpy as np
    import scipy.special
    def sellmeierkato2002z(x):
        return np.sqrt(4.59423+0.06206/((0.001*x)**2-0.04763)+110.80672/((0.001*x)**2-86.12171))
    def sellmeierkato2002y(x):
        return np.sqrt(3.45018+0.04341/((0.001*x)**2-0.04597)+16.98825/((0.001*x)**2-39.43799))
    def nsurffunc(x,a,b,c,d,f,g):
        return (a*1e-3) + (b*1e-6)*x + (c*1e-9)*x**2 + (d*1e-12)*x**3 + f*np.exp(-(x-350)/g)
    def ktpsurfz(x):
        return nsurffunc(x,26.7694730652534,-10.9737456325554,2.29268132448848,0,0.022459517196379,44.624769075701)
    def ktpsurfy(x):
        return nsurffunc(x,29.0815525865133,-6.58500255570493,2.13893700130814,0,0.00960546909966567,39.2004747033197)
    def erfc(x): # scaled so that x=1 at half-max i.e. erfc(1)==0.5
        return scipy.special.erfc(.47693627620447*x)
    nbulk = sellmeierkato2002z(λ) if 'z'==pol else sellmeierkato2002y(λ)
    nstep = conc*ktpsurfz(λ) if 'z'==pol else conc*ktpsurfy(λ)
    nair = 1 # print(f"nbulk,nstep,nair = {nbulk},{nstep},{nair}")
    return (nstep*erfc(-z/depth)*(np.abs(y)<width/2) + nbulk)*(z<=0) + nair*(z>0)
def ktpconcentration(x,d,c,p,r,exponential=False): # x=depth in µm, d = fwhm depth, c = surface conc, p = post-exchange anneal relative surface conc, r = reverse exchange depth
    # def erfc(x):
    #     return scipy.special.erfc(.47693627620447*x) # erfc==0.5 at x==1
    erfc = (lambda x: np.exp(-np.log(2)*x)) if exponential else (lambda x: scipy.special.erfc(.47693627620447*x)) # erfc==0.5 at x==1
    def f(x): # normal rb ion exchange
        return erfc(x/d)
    def fr(x): # reverse exchange
        return erfc(x/d) - erfc(x/abs(r))
    def fa(xfwhm,a,d0=5): # post exchange anneal
        # a = fraction of max index at surface, d0 = diffusion constant in um^2/hr
        xx,dx = np.linspace(0,50,501,retstep=True)
        ff = erfc(xx/xfwhm)
        ddff = np.zeros(len(ff))
        dt = 0.25 * dx**2 / d0 / 2
        time,steps = 0,0
        while a<ff[0]:
            dff = d0*np.diff(ff)
            ddff[0],ddff[1:-1],ddff[-1] = dff[0],dff[1:]-dff[:-1],-dff[-1]
            ff += ddff * dt / dx**2
            time,steps = time+dt,steps+1
        def func(x):
            return np.interp(x,xx,ff,left=0,right=0)
        return func
    assert 1==p or 0==r
    func = fr if r else fa(d,p) if p<1 else f
    return c*func(x)
class Ktpwaveguide(Anisowaveguide):
    @storecallargs
    def __init__(self,w=4,d=5,c=1,p=1,r=0,crystal='ktp',crystalstep='ktpsurf',sellcover='air',pol='v',cut='zyx',λ=None,bounds=None,step=0.2):
        self.w,self.d,self.c,self.p,self.r = w,d,c,p,r
        self.crystalstep,self.sellcover = crystalstep,sellcover
        bounds = bounds if bounds is not None else (-10,10,-30,2)
        super().__init__(crystal=crystal,pol=pol,cut=cut,λ=λ,bounds=bounds,deadbounds=None,step=step)
    def indexfunc(self,λ,axis):
        nbulk,nstep,nair = [index(λ,s) for s in (self.crystal+axis,self.crystalstep+axis,self.sellcover)]
        w,d,c,p,r = self.w,self.d,self.c,self.p,self.r
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        nn0 = nn.yslabs(ys=[0],ns=[nbulk,nair])
        nn1 = nn.yslabs(ys=[0],ns=[nbulk+nstep*ktpconcentration(-nn.yy,d,c,p,r),nair])
        return nn.xslabs(xs=[-0.5*w,0.5*w],ns=[nn0,nn1,nn0],debug=0)
    def nonpoled(self,Λ=1e9,λ1=1250,Δλ=50,λtol=1,verbose=False,**kwargs):
        return self.λqpm(Λ=Λ,λ1=λ1,λ2=None,Δλ=Δλ,λtol=λtol,verbose=verbose,Type='yzy',**kwargs)
class Ktpwaveguidesplit(Anisowaveguide):
    @storecallargs
    def __init__(self,w=4,d=5,c=1,p=1,r=0,split=0,crystal='ktp',crystalstep='ktpsurf',sellcover='air',pol='v',cut='zyx',λ=None,bounds=None,step=0.2):
        self.w,self.d,self.c,self.p,self.r,self.split = w,d,c,p,r,split
        self.crystalstep,self.sellcover = crystalstep,sellcover
        bounds = bounds if bounds is not None else (-20,20,-30,2)
        super().__init__(crystal=crystal,pol=pol,cut=cut,λ=λ,bounds=bounds,deadbounds=None,step=step)
    def indexfunc(self,λ,axis):
        nbulk,nstep,nair = [index(λ,s) for s in (self.crystal+axis,self.crystalstep+axis,self.sellcover)]
        w,d,c,p,r,split,g = self.w,self.d,self.c,self.p,self.r,self.split,self.split-self.w
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        nn0 = nn.yslabs(ys=[0],ns=[nbulk,nair])
        nn1 = nn.yslabs(ys=[0],ns=[nbulk+nstep*ktpconcentration(-nn.yy,d,c,p,r),nair])
        return nn.xslabs(xs=[-w-0.5*g,-0.5*g,0.5*g,0.5*g+w],ns=[nn0,nn1,nn0,nn1,nn0],debug=0) if split>w else nn.xslabs(xs=[-0.5*(w+split),0.5*(w+split)],ns=[nn0,nn1,nn0],debug=0)
class Ktpwaveguideexp(Ktpwaveguide):
    def indexfunc(self,λ,axis):
        nbulk,nstep,nair = [index(λ,s) for s in (self.crystal+axis,self.crystalstep+axis,self.sellcover)]
        w,d,c,p,r = self.w,self.d,self.c,self.p,self.r
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        nn0 = nn.yslabs(ys=[0],ns=[nbulk,nair])
        nn1 = nn.yslabs(ys=[0],ns=[nbulk+nstep*ktpconcentration(-nn.yy,d,c,p,r,exponential=1),nair])
        return nn.xslabs(xs=[-0.5*w,0.5*w],ns=[nn0,nn1,nn0],debug=0)
class Ktpwaveguideblock(Ktpwaveguide):
    def indexfunc(self,λ,axis):
        nbulk,nstep,nair = [index(λ,s) for s in (self.crystal+axis,self.crystalstep+axis,self.sellcover)]
        w,d,c,p,r = self.w,self.d,self.c,self.p,self.r
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        nn0 = nn.yslabs(ys=[0],ns=[nbulk,nair])
        nn1 = nn.yslabs(ys=[-d,0],ns=[nbulk,nbulk+nstep,nair])
        return nn.xslabs(xs=[-0.5*w,0.5*w],ns=[nn0,nn1,nn0],debug=0)
class Ktpwaveguidepadberg(Anisowaveguide):
    @storecallargs
    def __init__(self,w=4,c=1,t0=60,σ=1,k0=3.1,d0=0.25,crystal='ktp',crystalstep='ktpsurf',sellcover='air',pol='v',cut='zyx',λ=None,bounds=None,step=0.2):
        self.w,self.c,self.p,self.r,self.t0,self.σ,self.k0,self.d0 = w,c,1,0,t0,σ,k0,d0
        self.crystalstep,self.sellcover = crystalstep,sellcover
        bounds = bounds if bounds is not None else (-10,10,-30,2)
        super().__init__(crystal=crystal,pol=pol,cut=cut,λ=λ,bounds=bounds,deadbounds=None,step=step)
    def indexfunc(self,λ,axis):
        assert λ is not None
        nbulk,nstep,nair = [index(λ,s) for s in (self.crystal+axis,self.crystalstep+axis,self.sellcover)]
        w,c,t0,σ,k0,d0 = self.w,self.c,self.t0,self.σ,self.k0,self.d0
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        nn0 = nn.yslabs(ys=[0],ns=[nbulk,nair])
        nn1 = nn.yslabs(ys=[0],ns=[nbulk+nstep*padbergconcentration(np.clip(nn.xx,-w/2,w/2),-np.min(nn.yy,0),w=w,c=c,t0=t0,σ=σ,k0=k0,d0=d0,dy=stepy,y0=-y0),nair])
        return nn.xslabs(xs=[-0.5*w,0.5*w],ns=[nn0,nn1,nn0],debug=0)
@np.vectorize
def padbergconcentration(x,y,w,c,t0,dy,y0,σ=1,k0=3.1,d0=0.25,d={}):
    from scipy.special import erf
    key = (x,w,c,t0,dy,y0,σ,k0,d0)
    if key in d:
        return c * d[key](y,extrapolate='constant')
    def concdepdiffuse(t0=60,d0=0.25,k=0,z0=50,dz=0.1): # concentration dependent diffusion
        # t0 = diffusion time in min, d0 = diffusion constant in µm^2/min, z0,dz = grid size,step in µm
        from wavedata import logrounddown
        zz = wrange(0,z0,dz)
        ff,ddff = np.zeros(len(zz)),np.zeros(len(zz))
        dt = 0.25 * dz**2 / (2 * d0 * np.exp(k))
        # dt = logrounddown(dt)
        # for t in wrange(0,t0,dt):
        n = int(1+t0/dt)
        dt = t0/n
        for _ in range(n):
            ff[0] = 1
            dff = d0 * np.exp(k*0.5*(ff[:-1]+ff[1:])) * np.diff(ff)
            ddff[0],ddff[1:-1],ddff[-1] = dff[0],dff[1:]-dff[:-1],-dff[-1]
            ff += ddff * dt / dz**2
        ff[0] = 1
        return Wave(ff,zz)
    def f(x,w,σ=1): # Padberg2020
        return 1 + 0.5*erf((x-0.5*w)/σ) - 0.5*erf((x+0.5*w)/σ)
    d[key] = concdepdiffuse(t0=t0,k=k0*f(x,w,σ),z0=y0,dz=dy)
    return c * d[key](y,extrapolate='constant')
class Ktpstripwaveguide(Anisowaveguide):
    @storecallargs
    def __init__(self,w=4,d=5,c=1,p=0.9,r=0,sx=None,sy=0,crystal='ktp',crystalstep='ktpsurf',sellcover='air',sellstrip='sin',pol='v',cut='zyx',λ=None,bounds=None,step=0.2):
        self.w,self.d,self.c,self.p,self.r = w,d,c,p,r
        self.sx,self.sy = sx,sy
        self.crystalstep,self.sellcover,self.sellstrip = crystalstep,sellcover,sellstrip
        bounds = bounds if bounds is not None else (-10,10,-30,2)
        super().__init__(crystal=crystal,pol=pol,cut=cut,λ=λ,bounds=bounds,deadbounds=None,step=step)
    def indexfunc(self,λ,axis):
        nbulk,nstep,nair,nstrip = [index(λ,s) for s in (self.crystal+axis,self.crystalstep+axis,self.sellcover,self.sellstrip)]
        w,d,c,p,r,sx,sy = self.w,self.d,self.c,self.p,self.r,self.sx,self.sy
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        sx = sx if sx is not None else 2*(x1-x0)
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        nn0 = nn.xslabs(xs=[-0.5*w,0.5*w],ns=[nbulk,nbulk+nstep*ktpconcentration(-nn.yy,d,c,p,r),nbulk])
        nn1 = nn.xslabs(xs=[-0.5*sx,0.5*sx],ns=[nair,nstrip,nair])
        return nn.yslabs(ys=[0,sy],ns=[nn0,nn1,nair])
    def dmask(self):
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        dd = 1+0*Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy)) # dd = (dd.yy<=0)
        dd = inversestepfunction(dd.yy,0,stepx)
        return dd
class Ktpdoublestripwaveguide(Anisowaveguide): # KTP waveguide bonded to double SiN PIC platform
    @storecallargs
    def __init__(self,w=4,d=5,c=1,p=0.9,r=0,ay=0,bx=0,by=0,cy=0,dx=0,dy=0,crystal='ktp',crystalstep='ktpsurf',sellcover='sio2',sellstrip='sin',sellbuffer='sio2',pol='v',cut='zyx',λ=None,bounds=None,step=0.2):
        # layer thicknesses ay,by,cy,dy are cover,strip/cover,cover,stripcover with cover beyond, where ay is closest to KTP
        # layer widths bx,dx are width of strip (outside the width is cover), i.e. dx = first nitride width, bx = second nitride width
        self.w,self.d,self.c,self.p,self.r = w,d,c,p,r
        self.ay,self.bx,self.by,self.cy,self.dx,self.dy = ay,bx,by,cy,dx,dy
        self.crystalstep,self.sellcover,self.sellstrip,self.sellbuffer = crystalstep,sellcover,sellstrip,sellbuffer
        bounds = bounds if bounds is not None else (-10,10,-30,2)
        super().__init__(crystal=crystal,pol=pol,cut=cut,λ=λ,bounds=bounds,deadbounds=None,step=step)
    def indexfunc(self,λ,axis):
        nbulk,nstep,nair,nstrip,nbuf = [index(λ,s) for s in (self.crystal+axis,self.crystalstep+axis,self.sellcover,self.sellstrip,self.sellbuffer)]
        w,d,c,p,r = self.w,self.d,self.c,self.p,self.r
        ay,bx,by,cy,dx,dy = self.ay,self.bx,self.by,self.cy,self.dx,self.dy
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        nn0 = nn.xslabs(xs=[-0.5*w,0.5*w],ns=[nbulk,nbulk+nstep*ktpconcentration(-nn.yy,d,c,p,r),nbulk])
        nnb = nn.xslabs(xs=[-0.5*bx,0.5*bx],ns=[nair,nstrip,nair])
        nnd = nn.xslabs(xs=[-0.5*dx,0.5*dx],ns=[nair,nstrip,nair])
        return nn.yslabs(ys=[0,ay,ay+by,ay+by+cy,ay+by+cy+dy],ns=[nn0,nbuf,nnb,nair,nnd,nair])
    def dmask(self):
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        dd = 1+0*Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy)) # dd = (dd.yy<=0)
        dd = inversestepfunction(dd.yy,0,stepx)
        return dd
class Ktpsplitstripwaveguide(Anisowaveguide):
    @storecallargs
    def __init__(self,w=4,d=5,c=1,p=0.9,r=0,sx=0,sy=0,gx=0,crystal='ktp',crystalstep='ktpsurf',sellcover='sio2',sellstrip='sin',pol='v',cut='zyx',λ=None,bounds=None,step=0.2):
        # width of gap in strip is gx, combined width of two strips is sx (i.e. pair of strips each sx/2 wide separated by gx)
        self.w,self.d,self.c,self.p,self.r = w,d,c,p,r
        self.sx,self.sy,self.gx = sx,sy,gx
        self.crystalstep,self.sellcover,self.sellstrip = crystalstep,sellcover,sellstrip
        bounds = bounds if bounds is not None else (-10,10,-30,2)
        super().__init__(crystal=crystal,pol=pol,cut=cut,λ=λ,bounds=bounds,deadbounds=None,step=step)
    def indexfunc(self,λ,axis):
        nbulk,nstep,nair,nstrip = [index(λ,s) for s in (self.crystal+axis,self.crystalstep+axis,self.sellcover,self.sellstrip)]
        w,d,c,p,r,sx,sy,gx = self.w,self.d,self.c,self.p,self.r,self.sx,self.sy,self.gx
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        nn0 = nn.xslabs(xs=[-0.5*w,0.5*w],ns=[nbulk,nbulk+nstep*ktpconcentration(-nn.yy,d,c,p,r),nbulk])
        nn1 = nn.xslabs(xs=[-0.5*(sx+gx),-0.5*gx,0.5*gx,0.5*(sx+gx)],ns=[nair,nstrip,nair,nstrip,nair])
        return nn.yslabs(ys=[0,sy],ns=[nn0,nn1,nair])
    def dmask(self):
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        dd = 1+0*Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy)) # dd = (dd.yy<=0)
        dd = inversestepfunction(dd.yy,0,stepx)
        return dd
class Ktpsplitdoublestripwaveguide(Anisowaveguide): # KTP waveguide bonded to double SiN PIC platform
    @storecallargs
    def __init__(self,w=4,d=5,c=1,p=0.9,r=0,ay=0,bx=0,by=0,cy=0,dx=0,dy=0,gx=0,crystal='ktp',crystalstep='ktpsurf',sellcover='sio2',sellstrip='sin',sellbuffer='sio2',pol='v',cut='zyx',λ=None,bounds=None,step=0.2):
        # layer thicknesses ay,by,cy,dy are cover,strip/cover,cover,stripcover with cover beyond, where ay is closest to KTP
        # layer widths bx,dx are width of strip (outside the width is cover), i.e. dx = first nitride width, bx = second nitride width
        # width of gap in strip is gx, combined width of two strips is sx (i.e. pair of strips each sx/2 wide separated by gx)
        self.w,self.d,self.c,self.p,self.r = w,d,c,p,r
        self.ay,self.bx,self.by,self.cy,self.dx,self.dy,self.gx = ay,bx,by,cy,dx,dy,gx
        self.crystalstep,self.sellcover,self.sellstrip,self.sellbuffer = crystalstep,sellcover,sellstrip,sellbuffer
        bounds = bounds if bounds is not None else (-10,10,-30,2)
        super().__init__(crystal=crystal,pol=pol,cut=cut,λ=λ,bounds=bounds,deadbounds=None,step=step)
    def indexfunc(self,λ,axis):
        nbulk,nstep,nair,nstrip,nbuf = [index(λ,s) for s in (self.crystal+axis,self.crystalstep+axis,self.sellcover,self.sellstrip,self.sellbuffer)]
        w,d,c,p,r = self.w,self.d,self.c,self.p,self.r
        ay,bx,by,cy,dx,dy,gx = self.ay,self.bx,self.by,self.cy,self.dx,self.dy,self.gx
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        nn0 = nn.xslabs(xs=[-0.5*w,0.5*w],ns=[nbulk,nbulk+nstep*ktpconcentration(-nn.yy,d,c,p,r),nbulk])
        nnb = nn.xslabs(xs=[-0.5*(bx+gx),-0.5*gx,0.5*gx,0.5*(bx+gx)],ns=[nair,nstrip,nair,nstrip,nair])
        nnd = nn.xslabs(xs=[-0.5*(dx+gx),-0.5*gx,0.5*gx,0.5*(dx+gx)],ns=[nair,nstrip,nair,nstrip,nair])
        return nn.yslabs(ys=[0,ay,ay+by,ay+by+cy,ay+by+cy+dy],ns=[nn0,nbuf,nnb,nair,nnd,nair])
    def dmask(self):
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        dd = 1+0*Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy)) # dd = (dd.yy<=0)
        dd = inversestepfunction(dd.yy,0,stepx)
        return dd
from fipy import CellVariable,DiffusionTerm,TransientTerm,Grid2D
def lndiffusionsolver(width,depth,ape,rpe,ape2,gridx,gridy,resolution=0.1,timestep=0.5,zeroboundary=0,gap=0,xcut=0): # quiet = True
    def diffuse(C,hours=6,timestep=0.5,xcut=0):
        d_z,a_z,b_z,g_z = 0.414, 0.134, 34.5, 0.0497 # um^2/h Lenzini diffusion constants
        d_x,a_x,b_x,g_x = 0.334, 0.116, 30.7, 0.00711 # um^2/h Lenzini
        #d_z,a_z,b_z,g_z = .721, .08, 35, .065 # um^2/h Rosti
        Dx = d_x*(a_x + (1.-a_x)/(b_x*C+g_x))
        Dz = d_z*(a_z + (1.-a_z)/(b_z*C+g_z))
        diff = Dz.dot([[1,0],[0,0]]) + Dx.dot([[0,0],[0,1]]) if xcut else Dx.dot([[1,0],[0,0]]) + Dz.dot([[0,0],[0,1]])
        eqX = TransientTerm() == DiffusionTerm(diff)
        C.updateOld()
        remainingtime = hours
        while 0<remainingtime:
            step = timestep if timestep<remainingtime else remainingtime
            remainingtime,residual = remainingtime-step,1
            while residual>1e-3:
                residual = eqX.sweep(var=C,dt=step)
            C.updateOld() # if not quiet: print('\n',width,depth,ape,rpe,ape2,gridx,gridy,resolution,timestep,zeroboundary,gap,xcut,quiet)
    nx,ny = int(gridx/resolution+0.5),int(gridy/resolution+0.5)
    mesh = Grid2D(nx=nx,ny=ny,dx=resolution,dy=resolution)
    dxpe,dzpe = 0.098, 0.056 # pe diffusion in um^2/h
    undercut = 0.2*depth*(dzpe/dxpe)**.5 if xcut else 0.2*depth*(dxpe/dzpe)**.5 # if not quiet: print('undercut %.4f um' % (undercut))
    C = CellVariable(mesh,name='ape2d',value=0.,hasOld=1)
    X,Z = mesh.cellCenters
    np.seterr(invalid='ignore') # ignore NaNs from np.sqrt of negatives
    wdpe = (
        ( Z <= depth ) & ((
        ( X > resolution*nx/2. - gap/2. - width/2. - undercut*np.sqrt(1-np.square(Z/depth)) ) & 
        ( X < resolution*nx/2. - gap/2. + width/2. + undercut*np.sqrt(1-np.square(Z/depth)) ) ) | (
        ( X > resolution*nx/2. + gap/2. - width/2. - undercut*np.sqrt(1-np.square(Z/depth)) ) & 
        ( X < resolution*nx/2. + gap/2. + width/2. + undercut*np.sqrt(1-np.square(Z/depth)) ) )) )
    C.setValue(1,where = wdpe)
    if zeroboundary:
        C.constrain(0, where = mesh.facesTop|mesh.facesLeft|mesh.facesRight)
    if ape:
        diffuse(C,ape,timestep,xcut) # if not quiet: print('peak concentration after ape %.4f' % (np.max(C.value[None,:])))
    E = CellVariable(mesh,name='rpe2d',value=0.,hasOld=1)
    E.setValue(C)
    if rpe:
        E.constrain(0, where = mesh.facesBottom)
        if zeroboundary:
            E.constrain(0, where = mesh.facesTop|mesh.facesLeft|mesh.facesRight)
        diffuse(E,rpe,timestep,xcut) # if not quiet: print('peak concentration after rpe %.4f' % (np.max(E.value[None,:])))
    CC = CellVariable(mesh,name='ape2d2',value=0.,hasOld=1)
    CC.setValue(E)
    if ape2:
        if zeroboundary:
            CC.constrain(0, where = mesh.facesTop|mesh.facesLeft|mesh.facesRight)
        diffuse(CC,ape2,timestep,xcut) # if not quiet: print('peak concentration after ape2 %.4f' % (np.max(CC.value[None,:])))
    xs = np.linspace(-gridx/2+resolution/2,gridx/2-resolution/2,nx)
    ys = np.linspace(0+resolution/2,gridy-resolution/2,ny)
    ccr = CC.value[None,:]
    # return ccr.reshape(len(ys),len(xs)).T,xs,ys
    A,xs,ys = ccr.reshape(len(ys),len(xs)).T,xs,ys
    return Wave2D(A,xs,ys)
cachefolder = 'c:/backup/cache/lndiffusionsolvercache'
def lndiffusionargs(w,sa,a,r,a2,at,rt,a2t,split,*,bounds,resolution=0.1,timestep=0.5,zeroboundary=0,xcut=0,AE=1):
    def lnrescaletime(T1,T2,AE=AE): # T1,T2 in °C, AE in eV (best fit was AE=1eV)
        kB = 1/11604.51812 # eV/K
        return np.exp( AE/kB/(T1+273.15) - AE/kB/(T2+273.15) ) if T2 is not None else 1
    ape,rpe,ape2 = [x*lnrescaletime(328,temp) for x,temp in zip((a,r,a2),(at,rt,a2t))]
    (x0,x1,y0,y1) = bounds
    gridx,gridy,gap = 2*max(x1,-x0),-y0,split
    return tuple([float(x) for x in (w,sa,ape,rpe,ape2,gridx,gridy,resolution,timestep,zeroboundary,gap,xcut)])
def lndiffusionsolvercached(width,depth,ape,rpe=0,ape2=0,gridx=None,gridy=None,resolution=0.1,timestep=0.5,zeroboundary=0,gap=0,xcut=0):
    from diskcache import Cache
    cache = Cache(cachefolder)
    key = (width,depth,ape,rpe,ape2,gridx,gridy,resolution,timestep,zeroboundary,gap,xcut)
    if key not in cache:
        # print('cache miss',key)
        cache[key] = lndiffusionsolver(width,depth,ape,rpe,ape2,gridx,gridy,resolution,timestep,zeroboundary,gap,xcut)
    return cache[key]
class Rpewaveguide(Anisowaveguide): # z-cut waveguide
    @storecallargs
    def __init__(self,w=8,sa=1.9,a=23.5,r=24.5,a2=0,at=320,rt=300,a2t=None,split=0,bf=1,crystal='ln',crystalstep='lnzcutstep',sellcover='air',pol='v',cut='zyx',λ=None,bounds=None,step=0.2,diffres=0.1):
        self.polarizationcheck(pol)
        self.w,self.sa,self.a,self.r,self.a2,self.at,self.rt,self.a2t,self.split,self.bf,self.diffres = w,sa,a,r,a2,at,rt,a2t,split,bf,diffres
        self.crystal,self.crystalstep,self.sellcover = crystal,crystalstep,sellcover
        bounds = bounds if bounds is not None else (-15,15,-20,2)
        self.a2t = a2t if a2t is not None else at
        super().__init__(crystal=crystal,pol=pol,cut=cut,λ=λ,bounds=bounds,deadbounds=None,step=step)
    def polarizationcheck(self,pol):
        assert pol in 'vz', 'RPE z-cut only supports V polarization'
    def diffusionargs(self,w,sa,a,r,a2,at,rt,a2t,split,bounds,resolution):
        return lndiffusionargs(w,sa,a,r,a2,at,rt,a2t,split,bounds=bounds,resolution=resolution,xcut=0)
    def diffusioncompute(self):
        w,sa,a,r,a2,at,rt,a2t,split = self.w,self.sa,self.a,self.r,self.a2,self.at,self.rt,self.a2t,self.split
        # key = lndiffusionargs(w,sa,a,r,a2,at,rt,a2t,split,bounds=self.bounds)
        key = self.diffusionargs(w,sa,a,r,a2,at,rt,a2t,split,self.bounds,self.diffres)
        self.diffusionprofile_ = lndiffusionsolvercached(*key)
        return self
    def diffusionprofile(self):
        if not hasattr(self,'diffusionprofile_'):
            self.diffusioncompute()
        return self.diffusionprofile_
    def diffusionprecompute(self,num_cpus=None,**kwargs):
        from diskcache import Cache
        cache = Cache(cachefolder)
        w,sa,a,r,a2,at,rt,a2t,split = [kwargs.get(k,default) for k,default in zip(('w','sa','a','r','a2','at','rt','a2t','split'),(self.w,self.sa,self.a,self.r,self.a2,self.at,self.rt,self.a2t,self.split))]
        args = [w,sa,a,r,a2,at,rt,a2t,split]
        isiterable = [hasattr(x,'__len__') or isinstance(x,range) for x in args]
        Ls = [len(x) for i,x in zip(isiterable,args) if i]
        L = Ls[0] if len(Ls) else 1
        assert all([l==L for l in Ls]), f"All iterables must be same length. Lengths: {Ls}"
        args = [x if i else [x]*L for i,x in zip(isiterable,args)]
        # keys = [lndiffusionargs(*vals,bounds=self.bounds) for vals in zip(*args)]
        keys = [self.diffusionargs(*vals,bounds=self.bounds,resolution=self.diffres) for vals in zip(*args)]
        if not all([key in cache for key in keys]):
            import ray
            context = ray.init(num_cpus=num_cpus) # print('ray dashboard:',context.dashboard_url)
            @ray.remote
            def f(w,sa,ape,rpe,ape2,gridx,gridy,resolution,timestep,zeroboundary,gap,xcut):
                return lndiffusionsolver(w,sa,ape,rpe,ape2,gridx,gridy,resolution,timestep,zeroboundary,gap,xcut)
            futures = [] # import time; start = time.time()
            for key in keys:
                if key in cache:
                    futures.append(ray.put(cache[key]))
                else:
                    futures.append(f.remote(*key))
            results = ray.get(futures)
            ray.shutdown() # print('ray shutdown',time.time()-start,'sec')
            for key, result in zip(keys, results):
                if key not in cache:
                    cache[key] = result # print('cache miss',key,'key now in cache:',key in cache)
            print('diffusionprecompute complete')
        cache.close() # print('cache close',time.time()-start,'sec')
        return self
    def indexfunc(self,λ,axis):
        nbulk,nstep,nair = [index(λ,s) for s in (self.crystal+axis,self.crystalstep+axis,self.sellcover)]
        conc = self.diffusionprofile()
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        bf = self.bf
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        nn0 = nbulk + bf*nstep*conc(nn.xx,-nn.yy)
        return nn.yslabs(ys=[0],ns=[nn0,nair])
    def dmask(self):
        (x0,x1,y0,y1),(stepx,stepy),w,sa = self.bounds,self.step,self.w,self.sa
        dd = 1+0*Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        dd = inversestepfunction(dd.yy,0,stepx) - tophat(dd.xx,-0.5*w,0.5*w,stepx) * tophat(dd.yy,-sa,0,stepy)
        return dd

class Rpexcutwaveguide(Rpewaveguide): # x-cut waveguide
    @storecallargs
    def __init__(self,w=8,sa=1.9,a=23.5,r=24.5,a2=0,at=320,rt=300,a2t=None,split=0,bf=1,crystal='ln',crystalstep='lnxcutstep',sellcover='air',pol='h',cut='xzy',λ=None,bounds=None,step=0.2,diffres=0.1):
        super().__init__(w=w,sa=sa,a=a,r=r,a2=a2,at=at,rt=rt,a2t=a2t,split=split,bf=bf,crystal=crystal,crystalstep=crystalstep,sellcover=sellcover,pol=pol,cut=cut,λ=λ,bounds=bounds,step=step,diffres=diffres)
    def polarizationcheck(self,pol):
        assert pol in 'hx', 'RPE x-cut only supports H polarization'
    def diffusionargs(self,w,sa,a,r,a2,at,rt,a2t,split,bounds,resolution):
        return lndiffusionargs(w,sa,a,r,a2,at,rt,a2t,split,bounds=bounds,resolution=resolution,xcut=1)
class Tiwaveguide(Anisowaveguide):
    @storecallargs
    def __init__(self,w=4,d=0.1,a=5,atemp=995,split=0,bf=1,crystal='ln',sellcover='air',pol='v',cut='zyx',λ=None,bounds=None,step=0.2,diff='fouchet'):
        self.w,self.d,self.a,self.atemp,self.split,self.bf = w,d,a,atemp,split,bf
        self.sellcover,self.diff = sellcover,diff
        bounds = bounds if bounds is not None else (-30,30,-40,2)
        super().__init__(crystal=crystal,pol=pol,cut=cut,λ=λ,bounds=bounds,deadbounds=None,step=step)
    def concentration(self,unitless=True):
        def tidiffusionlength(t,T=1100,axis='z',diffmodel='fouchet'): # t in hrs, T in °C
            kB = 1/11604.51812 # eV/K
            D0,E0 = (1.35e8,2.22) if axis in 'xy' else (5e9,2.60)  # µm²/hr,eV # Fouchet1987
            if diffmodel=='strake':
                D0,E0 = (0.023*1e8*3600,30300*kB)               # µm²/hr,eV # Jain2010
            if diffmodel=='ganguly':
                D0 = 0.6232*np.exp(+E0/kB/(1050+273.15))        # µm²/hr # Ganguly1996
            if diffmodel=='parfenov':
                D0 = 0.19*np.exp(+E0/kB/(1000+273.15))          # µm²/hr # Parfenov2016
            D = D0*np.exp(-E0/kB/(T+273.15))
            return 2*np.sqrt(t*D)
        w,d,a,atemp,split,bf,diff,cut = self.w,self.d,self.a,self.atemp,self.split,self.bf,self.diff,self.cut
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        xcut = (cut[0] in 'xy')
        assert diff in ['fouchet','strake','ganguly','parfenov']
        # if 'parfenov'==diff and (λ!=1550 or not xcut): print('warning, parfenov only valid for 1550nm xcut')
        xd,yd = (tidiffusionlength(a,atemp,s,diff) for s in ('zy' if xcut else 'yz'))
        def c(x,y,width,depth,split=0):
            if not 0==split:
                if split<width:
                    return c(x,y,width=width+split,depth=depth)
                return c(x-split/2,y,width,depth) + c(x+split/2,y,width,depth)
            def cx(x,width):
                if 'parfenov'==diff:
                    return erf((0.5*width+x)/xd) + erf((0.5*width-x)/xd)
                return 0.5*erf((x+0.5*width)/xd) - 0.5*erf((x-0.5*width)/xd)
            def cy(y):
                if 'ganguly'==diff:
                    return 0.25*np.sqrt(np.pi)*(erf((y+depth)/yd)-erf((y-depth)/yd))
                if 'parfenov'==diff:
                    return erf((depth+y)/yd) + erf((depth-y)/yd)
                return depth/yd * np.exp(-y**2/yd**2)
                # return 0.25*np.sqrt(np.pi)*(erf((y+depth)/yd)-erf((y-depth)/yd)) if 'ganguly'==diff else np.exp(-y**2/yd**2)
            return cx(x,width) * cy(y) # strake88 eqs. 1 & 3, fouchet87 eqs. 10 & 11
        c0 = 1/1.57e-23 if 'strake'==diff else 3.8e22*2/np.sqrt(np.pi) # concentration constant in cm⁻³, note that c0 cancels out in the index calculation
        # note: ganguly 0.7072(2/√π)τ/d is nearly the same as fouchet 0.77τ/d surface concentration, so fouchet is used for ganguly concentration in cm⁻³
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        cc = c(nn.xx,nn.yy,w,d,split)
        return cc if unitless else (c0*cc,xd,yd) # units of cm⁻³, also return diffusion lengths if not unitless
    def indexfunc(self,λ,axis,legacy=False):
        nbulk,nair = index(λ,self.crystal+axis),index(λ,self.sellcover)
        w,d,a,atemp,split,bf,diff,cut = self.w,self.d,self.a,self.atemp,self.split,self.bf,self.diff,self.cut
        xcut = (cut[0] in 'xy')
        if 'parfenov'==diff and (λ!=1550 or not xcut): print('warning, parfenov only valid for 1550nm xcut')
        cc,_,yd = self.concentration(unitless=False) # note that c0 cancels out in the index calculation
        c0 = 1/1.57e-23 if 'strake'==diff else 3.8e22*2/np.sqrt(np.pi) # concentration constant in cm⁻³
        Fz,Fy,gammaz,gammay = ((1.2e-23,1.3e-25,1,0.55) if 'strake'==diff else 
                            # (1/c0,1/c0,0.765,0.5) if 'ganguly'==diff else # ganguly96 table 5 values
                            (1/c0,1/c0,0.877,0.561) if 'ganguly'==diff else # ganguly96 table 4 values
                            (1/c0,1/c0,0.8,0.5))
        dnz0 = index(λ,'lnstrakez') if 'strake'==diff else index(λ,'lnganguly0z')+index(λ,'lnganguly1z')*d/yd if 'ganguly'==diff else index(λ,'lnfouchet0z')+index(λ,'lnfouchet1z')*d/yd
        dny0 = index(λ,'lnstrakey') if 'strake'==diff else index(λ,'lnganguly0y')+index(λ,'lnganguly1y')*d/yd if 'ganguly'==diff else index(λ,'lnfouchet0y')+index(λ,'lnfouchet1y')*d/yd
        if 'parfenov'==diff:
            gammaz,gammay,dnz0,dny0 = 0.69,0.46,0.0419,0.0121
        dn = dnz0 * (Fz*cc)**gammaz if 'z'==axis else dny0 * (Fy*cc)**gammay
        if legacy:
            return (dn*bf + nbulk) * (cc.yy<=0) + nair * (0<cc.yy) # equivalent to Dielectric.tiwaveguide
        return cc.yslabs(ys=[0],ns=[dn*bf+nbulk,nair])
    def dmask(self):
        assert 0, 'not yet implemented'
class Tiwaveguidecustom(Anisowaveguide):
    @storecallargs
    def __init__(self,w=4,d=0.1,a=5,atemp=995,split=0,bf=1,crystal='ln',sellcover='air',pol='v',cut='zyx',λ=None,bounds=None,step=0.2,Dz=8.28e9,Ez=2.611,Dy=None,Ey=None,Fz=0.764,Fy=0.00828,γz=1,γy=0.55):
        self.w,self.d,self.a,self.atemp,self.split,self.bf = w,d,a,atemp,split,bf
        Dy,Ey = Dy if Dy is not None else Dz,Ey if Ey is not None else Ez
        self.Dz,self.Ez,self.Dy,self.Ey,self.Fz,self.Fy,self.γz,self.γy = Dz,Ez,Dy,Ey,Fz,Fy,γz,γy
        self.sellcover = sellcover
        bounds = bounds if bounds is not None else (-30,30,-40,2)
        super().__init__(crystal=crystal,pol=pol,cut=cut,λ=λ,bounds=bounds,deadbounds=None,step=step)
    def concentration(self):
        def tidiffusionlength(t,T,axis=None): # t in hrs, T in °C
            D0,E0 = (self.Dz,self.Ez) if 'z'==axis else (self.Dy,self.Ey)
            kB = 1/11604.51812 # eV/K
            D = D0*np.exp(-E0/kB/(T+273.15))
            return 2*np.sqrt(t*D)
        w,d,a,atemp,split,bf,cut = self.w,self.d,self.a,self.atemp,self.split,self.bf,self.cut
        (x0,x1,y0,y1),(stepx,stepy) = self.bounds,self.step
        xcut = (cut[0] in 'xy')
        xd,yd = (tidiffusionlength(a,atemp,s) for s in ('zy' if xcut else 'yz'))
        def c(x,y,width,split=0):
            if not 0==split:
                return c(x,y,width=width+split) if split<width else c(x-split/2,y,width) + c(x+split/2,y,width)
            def cx(x,width):
                return 0.5*erf((x+0.5*width)/xd) - 0.5*erf((x-0.5*width)/xd)
            def cy(y):
                return d/yd * np.exp(-y**2/yd**2)
            return cx(x,width) * cy(y)
        nn = Wave2D(xs=wrange(x0,x1,stepx),ys=wrange(y0,y1,stepy))
        cc = c(nn.xx,nn.yy,w,split)
        return cc
    def indexfunc(self,λ,axis):
        nbulk,nair,bf = index(λ,self.crystal+axis),index(λ,self.sellcover),self.bf
        Fz,Fy,γz,γy = self.Fz,self.Fy,self.γz,self.γy
        dnz0,dny0 = index(λ,'lnstrakez'),index(λ,'lnstrakey')
        cc = self.concentration()
        dn = dnz0 * (Fz*cc)**γz if 'z'==axis else dny0 * (Fy*cc)**γy
        return cc.yslabs(ys=[0],ns=[dn*bf+nbulk,nair])
class Tiwaveguidecustomfouchet(Tiwaveguidecustom):
    @storecallargs
    def __init__(self, w=4, d=0.1, a=5, atemp=995, split=0, bf=1, crystal='ln', sellcover='air', pol='v', cut='zyx', λ=None, bounds=None, step=0.2):
        if λ is None: raise ValueError("λ required to compute Fz,Fy.")
        Dz, Ez, Dy, Ey, γz, γy = 5e9, 2.60, 1.35e8, 2.22, 0.8, 0.5
        D, E = (Dy, Ey) if (cut[0] in 'xy') else (Dz, Ez)
        yd = 2 * np.sqrt(a * D * np.exp(-E * 11604.51812 /(atemp + 273.15)))
        Fz = ((index(λ, 'lnfouchet0z') + index(λ, 'lnfouchet1z') * (d/yd)) / index(λ, 'lnstrakez')) ** (1 / γz)
        Fy = ((index(λ, 'lnfouchet0y') + index(λ, 'lnfouchet1y') * (d/yd)) / index(λ, 'lnstrakey')) ** (1 / γy)
        super().__init__(w=w, d=d, a=a, atemp=atemp, split=split, bf=bf, crystal=crystal, sellcover=sellcover, pol=pol, cut=cut,
                         λ=λ, bounds=bounds, step=step, Dz=Dz, Ez=Ez, Dy=Dy, Ey=Ey, Fz=Fz, Fy=Fy, γz=γz, γy=γy)
class Tiwaveguidecustomganguly(Tiwaveguidecustom):
    @storecallargs
    def __init__(self, w=4, d=0.1, a=5, atemp=995, split=0, bf=1, crystal='ln', sellcover='air', pol='v', cut='zyx', λ=None, bounds=None, step=0.2):
        if λ is None: raise ValueError("λ required to compute Fz,Fy.")
        Dz, Ez, Dy, Ey, γz, γy = 4.987e9, 2.60, 1.78e8, 2.22, 0.877, 0.561
        D, E = (Dy, Ey) if (cut[0] in 'xy') else (Dz, Ez)
        yd = 2 * np.sqrt(a * D * np.exp(-E * 11604.51812 /(atemp + 273.15)))
        Fz = ((index(λ, 'lnganguly0z') + index(λ, 'lnganguly1z') * (d/yd)) / index(λ, 'lnstrakez')) ** (1 / γz)
        Fy = ((index(λ, 'lnganguly0y') + index(λ, 'lnganguly1y') * (d/yd)) / index(λ, 'lnstrakey')) ** (1 / γy)
        super().__init__(w=w, d=d, a=a, atemp=atemp, split=split, bf=bf, crystal=crystal, sellcover=sellcover, pol=pol, cut=cut,
                         λ=λ, bounds=bounds, step=step, Dz=Dz, Ez=Ez, Dy=Dy, Ey=Ey, Fz=Fz, Fy=Fy, γz=γz, γy=γy)
# FWHM index step depth = √(16 ln2 Dz t / γ) # https://chatgpt.com/share/68e67c54-8a7c-8007-a503-00c5e390b9d9
# peak Δn = index(λ,'lnstrakez') ( F d / √(4 Dz t) )^γ 
# Δn × FWHM index step depth = index(λ,'lnstrakez') √(16 ln2 Dz t / γ) ( F d / √(4 Dz t) )^γ 
# Δn × FWHM index step depth = index(λ,'lnstrakez') √(16 ln2 Dz / γ) ( F d / √(4 Dz) )^γ √t^(1-γ)

class Beamdata(collections.namedtuple('Beamdata','amplitude,ωx,ωy,x,y')):
    @property
    def center(self): return collections.namedtuple('Position','x y')(self.x,self.y)
    @property
    def ω(self): return np.sqrt(self.ωx*self.ωy)
    @property
    def mfdx(self): return 2*self.ωx
    @property
    def mfdy(self): return 2*self.ωy
    def couplingefficiency(self,bd,θ=0,λ=1064):
        T = 4/(self.ωx/bd.ωx + bd.ωx/self.ωx)/(self.ωy/bd.ωy + bd.ωy/self.ωy)
        if 0==θ: return T
        ωωx = 1/(1/self.ωx**2 + 1/bd.ωx**2)
        ωωy = 1/(1/self.ωy**2 + 1/bd.ωy**2)
        return T*np.exp( -0.5 * (2*np.pi*1000/λ)**2 * np.sqrt(ωωx*ωωy) * np.sin(θ)**2 )
def pmfiber(λ):
    fibers = [460,630,780,980,1550,2000]
    pmfibercutoff = {460:410, 630:570, 780:710, 980:920, 1300:1210, 1550:1380, 2000:2000} #, 2000:1720}
    pmfibercutoff = {**pmfibercutoff, '350':315, '405':380, '980':920}
    i = sum([pmfibercutoff[w]<λ for w in fibers])
    # assert i>0, 'no fiber with λcutoff < %s nm' % λ
    if i==0:
        # print('warning, %s fiber has λcutoff > %s nm' % (fibers[0],λ))
        return fibers[0]
    return fibers[i-1]
def fibermode(λ,a=None,l=0,x0=0,y0=0,res=0.1,limits=None,fiber=None,θx=0,θy=0): # λ in nm, a in µm
    def step(x,a0=0.5,dx=0): # returns 0 for x<0, 1 for 0<x, a0 is value at x==0
        if dx: return scipy.special.erfc(-x/dx)/2
        return np.heaviside(x,a0)

    #Corning
    fibers = [460,630,780,980,1550,2000]
    pmfibercutoff = {460:410, 630:570, 780:710, 980:920, 1300:1210, 1550:1380, 2000:2000} #, 2000:1720}
    pmfibercorediameter = {460:3.0, 630:3.5, 780:4.5, 980:5.5, 1300:8.0, 1550:8.5, 2000:7.0}
    pmcoreindex = {460:1.4697, 630:1.4617, 780:1.4587,980:1.4555,1550:1.449,2000:1.462}  # from Thorlabs tech support 11/13/18
    pmcladdingindex = {460:1.4648, 630:1.4571, 780:1.4537, 980:1.4507, 1550:1.444, 2000:1.449} # Δn = 0.005
    # pmindexwavelength = {460:532, 780:800, 980:1064, 1550:1550, 2000:1950}
    pmindexwavelength = {460:460, 630:630, 780:780, 980:980, 1550:1550, 2000:2000}
    pmmfd = {460:3.3, 630:4.5, 780:5.3, 980:6.6, 1300:9.3, 1400:9.8, 1550:10.1, 2000:8.0}
    pmmfdwavelength = {460:515, 630:630, 780:850, 980:980, 1300:1300, 1400:1450, 1550:1550, 2000:1950}
    ##PANDA PM fiber from Corning.pdf
    # fiber, lambda, MFD, beat length range (mm), 100m crosstalk (dB), λcutoff, attenuation (dB/km)
    # PM1550, 1550, 10.5 ± 0.5, 3.0-5.0, -30, 1300-1440, 0.5
    # PM14XX1400,-1490, 9.8 ± 0.5, 2.8-4.7, -30, 1260-1380, 1.0
    # PM1300, 1300, 9.0 ± 0.5, 2.5-4.0, -30, 1130-1270, 1.0
    # PM980, 980, 6.6 ± 0.5, 1.5-2.7, -30, 870-950, 2.5
    # PM850, 850, 5.5 ± 0.5, 1.0-2.0, -30, 650-800, 3.0
    # PM630, 630, 4.5 ± 0.5, ≤2.0, -30, 520-620, 12
    # PM480, 480, 4.5 ± 0.5, ≤2.0, -30, 400-470, 30
    # PM400, 410, 3.5 ± 0.5, ≤1.7, -30, 330-400, ≤50
    # for core diameter, effective NA, NA = √[Nco² - Ncl²], see: oz optics - optical fibers - DTS0135.pdf
    #Nufern
    # 350,405,460,630,780,850,980,1300,1400,1550,1950
    pmfibercutoff = {**pmfibercutoff, '350':315, '405':380, '980':920}
    pmfibercorediameter = {**pmfibercorediameter, '350':2.5, '405':3, '980':5.5}
    pmcoreindex = {**pmcoreindex, '350':None, '405':None, '980':None}
    pmcladdingindex = {**pmcladdingindex, '350':None, '405':None, '980':None}
    pmindexwavelength = {**pmindexwavelength, '350':350, '405':405, '980':980}
    pmmfd = {**pmmfd, '350':2.3, '405':3.3, '980':6.6}
    pmmfdwavelength = {**pmmfdwavelength, '350':350, '405':405, '980':980}
    # (fiber = '000' for Nufern, 000 for Corning)

    # note: fiber = '000' for Nufern, 000 for Corning
    n = index(λ,'sio2',20)
    fiber = fiber if fiber is not None else pmfiber(λ)
    assert fiber in fibers, f'PM{fiber} fiber not recognized'
    a = a if a is not None else pmfibercorediameter[fiber]/2
    if None in [pmcoreindex[fiber],pmcladdingindex[fiber],pmindexwavelength[fiber]]:
        ω = λ/pmmfdwavelength[fiber] * pmmfd[fiber]/2
        return gaussmode(λ,ω,ω,x0,y0,res,limits)
    # dn = 0.005
    dn = pmcoreindex[fiber] - pmcladdingindex[fiber]
    xmin,xmax,ymin,ymax = limits = limits if limits is not None else [-4*a,4*a,-4*a,4*a]
    V = (2*np.pi*1000/λ)*a*np.sqrt((n+dn)**2-n**2)
    ωapprox = a*(0.65 + 1.619/V**1.5 + 2.879/V**6)
    args = locals()
    from scipy.special import jv as J
    from scipy.special import kv as K
    # wx = Wave(index=np.linspace(0,5,51)); Wave().plots(*[J(n,wx) for n in [0,1,2,3]] + [K(n,wx) for n in [0,1,2,3]],ylim=(-1,2),groups=4)
    # https://www.rp-photonics.com/step_index_fibers.html # gloge1971 - Weakly Guiding Fibers.pdf
    # Ch 4, Step-index fibers (STU).pdf # https://www.hft.tu-berlin.de/fileadmin/fg154/ONT/Skript/ENG-Ver/STU_06_05.pdf
    def fl(b,l):
        u,v = V*np.sqrt(1-b),V*np.sqrt(b)
        return u*J(l+1,u)/J(l,u) - v*K(l+1,v)/K(l,v)
    def einside(r,b):
        return J(l,V*np.sqrt(1-b)*r/a) * (np.abs(r)<=a)
    def eoutside(r,b):
        x = K(l,V*np.sqrt(b)*np.abs(r)/a) * J(l,V*np.sqrt(1-b)*np.sign(r)) / K(l,V*np.sqrt(b)); #print('type(x)',type(x))
        x = np.nan_to_num(np.array(x))
        return x * (a<=np.abs(r))
    def eapprox(r):
        ω = a*(0.65 + 1.619/V**1.5 + 2.879/V**6)
        return np.exp(-r**2/ω**2)
    # wx = Wave(index=np.linspace(1e-9,1,1001)); Wave().plots(*[fl(wx,n) for n in [0,1,2]],ylim=(-100,100))
    xs = np.linspace(1e-9,1,1001)
    zs = np.diff(np.sign(fl(xs,l))) # diff(sign(w)) will be non-zero at cross-overs and will have the sign of the crossing: # https://stackoverflow.com/a/25091643
    zerocrossings = np.where(-2==zs)[0] # eg. zs = [0 0 0 0 0 0 0 0 2 0 0 0 -2 0 0 0]
    # args['zerocrossings'] = zerocrossings # print('approximate zeros at',[xs[z] for z in zerocrossings]) # + to - are the crossings we want, - to + are -∞ to +∞
    # wx = Wave(index=np.linspace(1e-9,xs[zerocrossings[3]],101)); Wave().plots(fl(wx,0),0*wx,pause=0)
    # if plotzerocrossings: wx = Wave(index=xs); Wave().plots(fl(wx,0),0*wx,ylim=(-100,100),pause=0)
    if 0==len(zerocrossings): assert 0,'no modes found, try l=0'
    z0 = zerocrossings[-1] # last crossing is the fundamental mode, others are higher order modes
    b0 = scipy.optimize.brentq(lambda b:fl(b,l), xs[z0], xs[z0+1])
    neff = b0*dn+n
    # if plot: wx = Wave(index=np.linspace(-2*a,2*a,101)); Wave().plots(einside(wx,b0),eoutside(wx,b0),eapprox(wx),pause=0)#,ylim=(None,1e-2))
    # make 2D data
    # xx,yy = Wave2D(xs=np.arange(xmin,xmax+res/2,res),ys=np.arange(ymin,ymax+res/2,res),returngrid=1)
    resx,resy = res if hasattr(res,'__len__') else [res,res]
    xx,yy = Wave2D(xs=np.arange(xmin,xmax+resx/2,resx),ys=np.arange(ymin,ymax+resy/2,resy)).grid()
    rr,θθ = np.sqrt((xx-x0)**2+(yy-y0)**2),np.arctan2(yy-y0,xx-x0)
    nn = n + dn*step(a-rr,dx=min(resx,resy)) # nn.plot()
    ee = (step(a-rr)*einside(rr,b0) + (1-step(a-rr))*eoutside(rr,b0)) * np.cos(l*(θθ-np.pi/2)) # ee.plot(legendtext='fibermode ee',pause=1)
    if θx or θy:
        ee = ee * exp(1j*2000*pi*θx*xx/λ) * exp(1j*2000*pi*θy*yy/λ)
    # md = Modedata(λ,'sio2',[neff],nn,n,dn,[ee],0,args) # md.ex.plot()
    md = Modedata(λ,[neff],nn,nn,nn,[ee],[0*ee],[0*ee],[0*ee],[0*ee],[0*ee]) # md.ex.plot()
    # λ,neffs,nx,ny,nz,Exs,Eys,Ezs,Hxs,Hys,Hzs
    return md
def gaussmode(λ=None,ωx=None,ωy=None,x0=0,y0=0,res=0.1,limits=None):
    ωy = ωy if ωy is not None else ωx
    xmin,xmax,ymin,ymax = limits = limits if limits is not None else [x0-4*ωx,x0+4*ωx,y0-4*ωx,y0+4*ωx] # exp(-4**2) = 1e-7
    args = locals()
    resx,resy = res if hasattr(res,'__len__') else [res,res]
    # xx,yy = Wave2D(xs=np.arange(xmin,xmax+res/2,res),ys=np.arange(ymin,ymax+res/2,res),returngrid=1)
    xx,yy = Wave2D(xs=np.arange(xmin,xmax+resx/2,resx),ys=np.arange(ymin,ymax+resy/2,resy)).grid()
    ee = np.exp( -((xx-x0)/ωx)**2 -((yy-y0)/ωy)**2 ) # e-field not intensity
    md = Modedata(λ,'air',[1],1+0*ee,1,0,[ee],0,args) # 'λ,sell,neffs,nn,nsub,dns,ees,mode,kwargs'
    return md

if __name__ == '__main__':
    def simpletest():
        wg = Boxwaveguide(1,1,1.8,λ=1064) # wg.plot()
        md = wg.modesolve() # md.plot()
        print(f"neff={md.neff:g}, Δn={md.dneff:g}")
        qd0 = wg().qpm(1560)
        print(f"Λ={qd0.Λ:g}")
        qd1 = wg(sell='ktpz').qpm(1064)
        print(f"η={qd1.sfgce():g}%/W/cm²")
        assert np.allclose(md.neff,1.63906), md.neff
        assert np.allclose(qd0.Λ,4.037450), qd0.Λ
        assert np.allclose(qd1.Λ,4.820896), qd1.Λ
        assert np.allclose(qd1.sfgce(),38748.4), qd1.sfgce()
        # neff=1.63906, Δn=0.639055 # Λ=4.03745 # η=38748.4%/W/cm²
    simpletest()
    from wavedata import timeit
    @timeit
    def test():
        if 0:
            wg = Stepfiberwaveguide(λ=1550,r=0.5*8.5,ncore=1.449,nclad=1.444,pol='v',sell=None,bounds=None,step=0.2) # Corning PM1550 fiber # wg.plot()
            md = wg.modesolve(verbose=0); print(f"neff={md.neff:g}, Δn={md.dneff:g}")
            md = wg.solve(verbose=0); print(f"neff={md.neff:g}, Δn={md.dneff:g}")
            md = wg.modesolve(method='exact',nummodes=2,verbose=0); print(f"neff={md.neff:g}, Δn={md.dneff:g}")
            md = wg.solve(method='exact',nummodes=2,verbose=0); print(f"neff={md.neff:g}, Δn={md.dneff:g}")
            md = Ktpwaveguide(λ=1064,pol='h',step=0.5).modesolve(method=None,verbose=0); print(f"\nneff={md.neff:g}, Δn={md.dneff:g}")
            md = Ktpwaveguide(λ=1064,pol='h',step=0.5).solve(method='supress',verbose=0); print(f"neff={md.neff:g}, Δn={md.dneff:g}")
            md = Ktpwaveguide(λ=1064,pol='v',step=0.5).modesolve(method='exact',verbose=0); print(f"\nneff={md.neff:g}, Δn={md.dneff:g}")
            md = Ktpwaveguide(λ=1064,pol='v',step=0.5).solve(method='exact',verbose=0); print(f"neff={md.neff:g}, Δn={md.dneff:g}")
            md = Ktpwaveguide(λ=532,w=4,d=2,pol='v',step=0.5).modesolve(method='isotropic',nummodes=2,mode=0,verbose=0); print(f"\nneff={md.neff:g}, Δn={md.dneff:g}")#; print(f"dneffs",md.dneffs)
            md = Ktpwaveguide(λ=532,w=4,d=2,pol='v',step=0.5).solve(method='isotropic',nummodes=4,mode=0,verbose=0); print(f"neff={md.neff:g}, Δn={md.dneff:g}")#; print(f"dneffs",md.dneffs)
            print('\nmd.Lc',md.Lc,'mm')
            md = Ridgewaveguide(2,2,1,step=0.5,λ=1064,pol='v',bounds=(-2,2,-2,2)).modesolve(method='exact',nummodes=20,verbose=0); print(f"\nneff={md.neff:g}, Δn={md.dneff:g}")
            md = Ridgewaveguide(2,2,1,step=0.5,λ=1064,pol='v',bounds=(-2,2,-2,2)).solve(method='exact',nummodes=20,verbose=0); print(f"neff={md.neff:g}, Δn={md.dneff:g}")
            print('poyntingvector',md.poyntingvector('z').max())
            print(Modedata.nan()) # 24s total
            # md = Ktpwaveguide(λ=1550,pol=None,step=0.5).solve(nummodes=300,method='exact'); print(md,md.neffs[:3],md.neffs[-3:]); print(md.pols().upper(), md.modecount()) # 47s
            # mdv = md.filterpolarization('v'); print(mdv.pols().upper(),mdv.modecount())
            # mdh = md.filterpolarization('h'); print(mdh.pols().upper(),mdh.modecount())
            # md = Ktpwaveguide(λ=1064,pol='h',step=0.5).solve(mode=4,nummodes=3,method='exact',verbose=1); print(md,md.neffs[-1:]); print(md.pols().upper()) # 630s
        # md = Ktpwaveguide(λ=1064,pol='h',step=0.5).modesolve(method=None,verbose=0); print(f"\nneff={md.neff:g}, Δn={md.dneff:g}")
        md = Rpewaveguide(λ=1550).modesolve(); print(f"Δn={md.dneff:g} MFDx={md.mfdx:g} MFDy={md.mfdy:g}") # Δn=0.00597764 MFDx=6.58309 MFDy=4.4354
        md = Rpewaveguide(λ=1550,bf=0.5).modesolve(); print(f"Δn={md.dneff:g} MFDx={md.mfdx:g} MFDy={md.mfdy:g}") # Δn=0.00135151 MFDx=9.35576 MFDy=6.54403
        # md = Rpewaveguide(λ=1550).diffusionprecompute().modesolve(); print(f"Δn={md.dneff:g} MFDx={md.mfdx:g} MFDy={md.mfdy:g}")
        md = Rpewaveguide(λ=1550).diffusionprecompute(w=[1,2,3,4,5,6]).modesolve(); print(f"Δn={md.dneff:g} MFDx={md.mfdx:g} MFDy={md.mfdy:g}")
        wg = Rpexcutwaveguide(λ=638,w=4,sa=0.7,a=16,r=2,at=300,rt=300,crystal='mgln',bounds=(-30,30,-40,2),step=0.2)
        md = wg.modesolve(); print(f"Δn={md.dneff:g} MFDx={md.mfdx:g} MFDy={md.mfdy:g}") # Δn=0.0128185 MFDx=2.41538 MFDy=1.42888
        wg = Rpexcutwaveguide(λ=1550,w=6,sa=0.9,a=4,r=0,a2=0,at=388,crystalstep='lnxcutstep',sellcover='air',pol='h',cut='xzy',bounds=None,step=0.2,diffres=0.1)
        [wg.diffusionprecompute(a=apes,sa=sa,w=w) for sa in sas for w in track(widths)]
        [wg(a=a,sa=sa,w=w).modesolve() for sa in sas for w in track(widths) for a in apes]
        md = wg.modesolve();print(f"n={md.neff:g} Δn={md.dneff:g} MFDx={md.mfdx:g} MFDy={md.mfdy:g}")
    # test()
    wg = Boxwaveguide(1,1,1.57,λ=1064) # wg.plot()
    md = wg.modesolve() # md.plot()

    ktp2d(0,0,1550,conc=0.5,pol='z')
    ktp2d(0,0,1550,conc=0.5,pol='y')



    # Type II qpm check ridge
    # pol,cut in modedata? or is sell+axis only needed for qpmdata
    # deadbounds (and etchbraggoverlap)
    # move fibermodes
    # directional couplers, shift and combine
    # __str__ for wg,md,qd
    # zhu
    # fix ridgewgtest
    # fix ktpwgtest(extended=1)
    # ln
    # trapezoid ridge
    # rotate
    # wg.plot with mode
    # wg.plot with outline
    # ridge y test fail
    # sfg design document
    # wdm design document
    # convert modesolver to python
    # eliminate support for plural modes in Modedata0
    # eliminate kwargs from Modedata0?

    # Modedata redesign:
    # - how to get dielectric args into md
    # - how to get qpm from md
    # - fix H,V ridge issue for method=exact,None
    # - Dielectric returns md list for any list in args? Use decorator, np.vectorize?
