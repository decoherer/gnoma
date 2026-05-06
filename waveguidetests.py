
import numpy as np
from waveguide import Waveguide,Planewaveguide,Stepfiberwaveguide,Corningpmfiberwaveguide
from waveguide import Qpmdata,Ridgewaveguide,Trapezoidalridgewaveguide,Ktpwaveguide,Ktpstripwaveguide,Sinwaveguide
from wavedata import Wave,Wave2D,wrange,timeit,track
from numpy import pi,nan
W = Wave

def simplesttest(w=2,h=1,n=1.5,dx=0.2,plot=0):
    nn = Wave2D(xs=wrange(-w/2-1,w/2+1,dx),ys=wrange(-h/2-1,h/2+1,dx))
    nn = 1 + (n-1)*nn.centeredrectangle(w,h)
    wg = Waveguide(nn,pol='v',λ=1064)
    if plot: wg.plot()
    md0 = wg.modesolve()
    print(' md0.neff',md0.neff,'md0.mfdx',md0.mfdx,'md0.mfdy',md0.mfdy)
    if plot: md0.plot()
    print(' wg.callargs',wg.callargs.keys())
    md1 = wg().modesolve()
    print(' md1.neff',md1.neff,'md1.mfdx',md1.mfdx,'md1.mfdy',md1.mfdy)
    md2 = wg(λ=532).modesolve()
    print(' md2.neff',md2.neff,'md2.mfdx',md2.mfdx,'md2.mfdy',md2.mfdy)
def wgtest(pol,cut):
    print('****pol,cut',pol,cut)
    wg = Ridgewaveguide(λ=1064,pol=pol,cut=cut)
    wg.plot(show=0)
    md = wg.modesolve() # md.plot()
    print(' md.pmfiber',md.pmfiber())
    print(' md.fibercoupling',md.fibercoupling())
    print(' md.dnmax',md.dnmax)
    print(' wg.callargs',wg.callargs)
    print(' md.mfdx',md.mfdx,'md.mfdy',md.mfdy)
    md1 = wg(λ=wg.λ/2).modesolve()
    print(' md1.neff',md1.neff,'md1.mfdx',md1.mfdx,'md1.mfdy',md1.mfdy,'md1.λ',md1.λ)
def ridgeshgtest(verbose=True):
    md1 = Ridgewaveguide(λ=1064,pol='h',cut='zyx').modesolve() # md1.plot()
    md2 = Ridgewaveguide(λ=1064,pol='v',cut='zyx').modesolve() # md2.plot()
    md3 = Ridgewaveguide(λ= 532,pol='h',cut='zyx').modesolve() # md3.plot()
    md4 = Ridgewaveguide(λ= 532,pol='v',cut='zyx').modesolve()
    qd = Qpmdata(md1,md2,md3) # print('qd.Λ',qd.Λ)
    qd = Qpmdata(md2,md2,md4) # print('qd.Λ',qd.Λ)
    wg = Ridgewaveguide(crystal='ln',step=0.2)
    md1 = wg(λ=1064,pol='h').modesolve()
    md2 = wg(λ=1064,pol='v').modesolve()
    md3 = wg(λ= 532,pol='h').modesolve()
    md4 = wg(λ= 532,pol='v').modesolve()
    qd0 = Qpmdata(md1,md2,md3)
    qd1 = wg.qpm(1064,1064)
    md22,md44 = [md.modesolve() for md in wg(λ=[1064,532],pol=['v','v'])]
    qd2 = Qpmdata(md22,md22,md44)
    qd = Qpmdata(md2,md2,md4)
    md5 = wg(λ=1064,pol='h').modesolve(mode=3,nummodes=4) # md5.plot()
    try:
        modeids = ' '.join([md.identifymode(asstring=1) for md in md5])
        assert modeids=='00 01 10 11',modeids
    except:
        print('md.identifymode failed')
        modeids = []
    if verbose:
        print()
        print('md1.sell',md1.sell,'md2.sell',md2.sell,'md3.sell',md3.sell)
        print('Λ0',qd0.Λ,'Λ',qd1.Λ,'ηsfg',qd1.sfgce(),'*')
        print('Λ0',qd0.Λ,'Λ',qd2.Λ,'ηsfg',qd2.sfgce())
        print('Λ0',qd0.Λ,'Λ',qd.Λ,'ηsfg',qd.sfgce())
        print('*'+str(qd)+'*')
        print('modeids',modeids)
        print('*'+str(md5)+'*')
        print()
    # print('wg',wg)
    assert md1.sell=='lny' and md2.sell=='lnz' and md3.sell=='lny', md1.sell+md2.sell+md3.sell
    assert np.allclose(qd0.Λ, 4.081569290366661, rtol=1e-2),qd0.Λ
    assert np.allclose(qd1.Λ, 6.734031823677308, rtol=1e-2), qd1.Λ
    assert np.allclose(qd.Λ, 6.734031823690504, rtol=1e-2), qd.Λ
    assert np.allclose(qd.sfgce(), 634.1774074558135, rtol=1e-2), qd.sfgce()
    # assert str(qd)=='Qpmdata(md1=  ► 1064λ lnz mfdx07.1 mfdy07.1 Δn0.705, md2=  ► 1064λ lnz mfdx07.1 mfdy07.1 Δn0.705, md3=  ►  532λ lnz mfdx07.1 mfdy07.0 Δn0.773, args={})',str(qd)
    assert str(qd)=='Qpmdata(md1=  ► 1064λ lnz mfdx07.0 mfdy07.0 Δn0.705, md2=  ► 1064λ lnz mfdx07.0 mfdy07.0 Δn0.705, md3=  ►  532λ lnz mfdx06.9 mfdy06.9 Δn0.773, args={})',str(qd)
    assert str(md5)=='  ► 1064λ lny mfdx03.3 mfdy03.3 Δn0.777',str(md5)
def ridgewgtest():
    from modes import modesolver
    md = modesolver(1064,10,10,5,0,'mglnridgez',res=0.5); assert np.allclose(md.dneff,0.6960530237770355,rtol=1e-4), md.dneff
    md = modesolver(1064,10,10,5,0,'mglnridgez',res=0.5,mode=0,nummodes=2,method='isotropic'); assert np.allclose(md.dneff,0.6959977153270969,rtol=1e-4), md.dneff
    md = modesolver(1064,10,10,5,0,'mglnridgez',res=0.5,method='zhu'); assert np.allclose(md.dneff,0.6960530237766824,rtol=1e-4), md.dneff
    md = modesolver(1064,10,10,5,0,'mglnridgey',res=0.5); assert np.allclose(md.dneff,0.6960015847993608,rtol=1e-4), md.dneff
    md = modesolver(1064,10,10,5,0,'mglnridgey',res=0.5,mode=0,nummodes=2,method='isotropic'); assert np.allclose(md.dneff,0.6960015972826485,rtol=1e-4), md.dneff
    md = modesolver(1064,10,10,5,0,'mglnridgey',res=0.5,method='zhu'); assert np.allclose(md.dneff,0.696001584799407,rtol=1e-4), md.dneff
    # return
    wg = Ridgewaveguide(10,10,5,step=0.5,λ=1064,pol='v')
    md = wg(pol='v').modesolve(verbose=0); assert np.allclose(md.dneff,0.6960530237770355,rtol=1e-4), md.dneff
    md = wg(pol='v').modesolve(nummodes=2,method='isotropic',verbose=0); assert np.allclose(md.dneff,0.6959977153270969,rtol=1e-4), md.dneff
    md = wg(pol='v').solve(solver='zhu',boundary=None,verbose=0); assert np.allclose(md.dneff,0.696157,rtol=1e-4), md.dneff
    assert np.allclose(md.dneff,0.6960530237766824,rtol=1e-3), md.dneff
    md = wg(pol='h').modesolve(); assert np.allclose(md.dneff,0.77793,rtol=1e-3), md.dneff
    md = wg(pol='h').modesolve(nummodes=2,method='isotropic'); assert np.allclose(md.dneff,0.77793,rtol=1e-3), md.dneff
    md = wg(pol='h').solve(solver='zhu'); assert np.allclose(md.dneff,0.77793,rtol=1e-3), md.dneff
def ktpwgtest(extended=False):
    mdv = Ktpwaveguide(λ=1064,pol='v',step=0.2).modesolve()
    assert np.allclose(mdv.dneff,0.00796283090156269), mdv.dneff
    assert np.allclose(mdv.mfdx,4.378776836518622), mdv.mfdx
    assert np.allclose(mdv.mfdy,4.745639104361813,atol=1e-3), mdv.mfdy
    mdh = Ktpwaveguide(λ=1064,pol='h',step=0.2).modesolve() # mdh.plotindex()
    assert np.allclose(mdh.dneff,0.012745084078242153), mdh.dneff
    assert np.allclose(mdh.mfdx,4.0831639474890915), mdh.mfdx
    assert np.allclose(mdh.mfdy,4.123692761399877,atol=1e-3), mdh.mfdy # 4.123937070856186
    wg = Ktpwaveguide(λ=1064,pol='v',step=0.2)
    qd = wg.qpm(1064)
    assert np.allclose(qd.Λ,8.020173882321052), qd.Λ
    # qda = qpm(1064,1064,width=4,depth=5,ape=1.0,rpe=0.9,sell='ktp',Type='yzy',res=0.2) # print(qda.Λ) # 54.67409532682693
    qdb = wg.qpm(1064,Type='hvh') # print(qdb.Λ,qdb.Λ-qda.Λ) # 55.12456277564374 0.4504674488168092
    assert np.allclose(qdb.Λ,55.1245627796058), qdb.Λ
    qdc = wg.qpm(1064,Type='yzy')
    assert np.allclose(qdc.Λ,55.1245627796058), qdc.Λ
    qdd = wg.qpm(1064,Type='yyz')
    assert np.allclose(qdd.Λ,3.6500496282564945), qdd.Λ
    # # res comparison
    # from modes import qpm
    # qdx = qpm(1550,810,width=3,depth=5,ape=1.0,rpe=0.9,sell='ktp',Type='zzz',res=0.2); print(qdx.Λ) # 8.609033727614284
    # qdy = Ktpwaveguide(3,5,1,0.9,step=(0.2,0.02)).qpm(1550,810); print(qdy.Λ,qdy.Λ-qdx.Λ) # 8.613493870085385 0.004460142471101491
    # qdx = qpm(1550,810,width=3,depth=5,ape=1.0,rpe=0.9,sell='ktp',Type='zzz',res=0.1); print(qdx.Λ) # 8.61114145614161
    # qdy = Ktpwaveguide(3,5,1,0.9,step=(0.1,0.02)).qpm(1550,810); print(qdy.Λ,qdy.Λ-qdx.Λ) # 8.613283866905864 0.0021424107642538104
    # qdx = qpm(1550,810,width=3,depth=5,ape=1.0,rpe=0.9,sell='ktp',Type='zzz',res=0.05); print(qdx.Λ) # 8.61164521502637
    # qdy = Ktpwaveguide(3,5,1,0.9,step=(0.05,0.02)).qpm(1550,810); print(qdy.Λ,qdy.Λ-qdx.Λ) # 8.612823147640821 0.0011779326144516489
    # qdx = qpm(1550,810,width=3,depth=5,ape=1.0,rpe=0.9,sell='ktp',Type='zzz',res=0.02); print(qdx.Λ) # 8.612061498696601
    # qdy = Ktpwaveguide(3,5,1,0.9,step=(0.02,0.02)).qpm(1550,810); print(qdy.Λ,qdy.Λ-qdx.Λ) # 8.612683146558497 0.000621647861896335
    if extended:
        ...
        # # from modes import modesolver
        # # md = modesolver(1064,4,5,1,0.9,sell='ktpy',res=0.5,mode=0,nummodes=2,verbose=False,cachelookup=0); print(md.dneff) # 0.013118785656799137
        # # md = modesolver(1064,4,5,1,0.9,sell='ktpy',res=0.5,mode=0,nummodes=2,verbose=False,method='isotropic',cachelookup=0); print(md.dneff) # 0.013136920181510359
        # # md = modesolver(1064,4,5,1,0.9,sell='ktpz',res=0.5,mode=0,nummodes=2,verbose=False,method='isotropic',cachelookup=0); print(md.dneff) # 0.008463012796738889
        # # md = modesolver(1064,4,5,1,0.9,sell='ktpy',res=0.2,mode=0,nummodes=2,verbose=False,cachelookup=0); print(md.dneff) # 0.013071992173811164
        # # md = modesolver(1064,4,5,1,0.9,sell='ktpy',res=0.2,mode=0,nummodes=2,verbose=False,method='isotropic',cachelookup=0); print(md.dneff) # 0.013089939772577885
        # # md = modesolver(1064,4,5,1,0.9,sell='ktpz',res=0.2,mode=0,nummodes=2,verbose=False,method='isotropic',cachelookup=0); print(md.dneff) # 0.008383789944844677

        # md0 = Ktpwaveguide(4,5,p=0.9,λ=1064,pol='v',step=0.5).modesolve() # print(md0.dneff) # 0.007597696859860337
        # # md1 = Ktpwaveguide(4,5.25,c=5.25/5,p=0.9,λ=1064,pol='v',step=0.5).modesolve() # print(md1.dneff) # 0.008388608925088192
        # # md2 = modesolver(1064,4,5,1,0.9,sell='ktpz',res=0.5,verbose=False,cachelookup=0) # print(md2.dneff) # 0.008269580415563693
        # md3 = Ktpwaveguide(4,5,p=1,λ=1064,pol='v',step=0.5).modesolve() # print(md3.dneff) # 0.007616587249267726
        # # md4 = Ktpwaveguide(4,5.25,c=5.25/5,p=1,λ=1064,pol='v',step=0.5).modesolve() # print(md4.dneff) # 0.008422391948285535
        # # md5 = modesolver(1064,4,5,1,0,sell='ktpz',res=0.5,verbose=False,cachelookup=0) # print(md5.dneff) # 0.008385264439793927
        # md6 = Ktpwaveguide(4,5.0,p=1,λ=1064,pol='v',step=0.1).modesolve() # print(md6.dneff) # 0.008101824953729686
        # # md7 = Ktpwaveguide(4,5.1,p=1,λ=1064,pol='v',step=0.1).modesolve() # print(md7.dneff) # 0.008187347329445371
        # # md8 = Ktpwaveguide(4,5.05,c=5.05/5,p=1,λ=1064,pol='v',step=0.1).modesolve() # print(md8.dneff) # 0.008265520850355701
        # # md9 = modesolver(1064,4,5,1,0,sell='ktpz',res=0.1,verbose=False,cachelookup=0) # print(md9.dneff) # 0.008208867521978691
        # mda = Ktpwaveguide(4,5.00,p=1,λ=1064,pol='v',step=0.05).modesolve() # print(mda.dneff) # 0.008129102684824252
        # # mdb = Ktpwaveguide(4,5.025,p=1,λ=1064,pol='v',step=0.05).modesolve() # print(mdb.dneff) # 0.008150642495925009
        # # mdc = Ktpwaveguide(4,5.025,c=5.025/5,p=1,λ=1064,pol='v',step=0.05).modesolve() # print(mdc.dneff) # 0.008210987161993977
        # # mdd = modesolver(1064,4,5,1,0,sell='ktpz',res=0.05,verbose=False,cachelookup=0) # print(mdd.dneff) # 0.008175464382250253

        # assert np.allclose(md0.dneff,0.007597696859860337), md0.dneff
        # assert np.allclose(md3.dneff,0.007616587249267726), md3.dneff
        # assert np.allclose(md6.dneff,0.008101824953729686), md6.dneff
        # assert np.allclose(mda.dneff,0.008129102684824252), mda.dneff

        # from modes import qpm
        # # qd0 = qpm(1550,810,width=3,depth=5,ape=1.0,rpe=0.9,sell='ktp',Type='zzz',res=0.1) # print(qd0.Λ)
        # qd1 = Ktpwaveguide(3,5,1,0.9,step=0.1).qpm(1550,810) # print(qd1.Λ,qd1.Λ-qd0.Λ)
        # assert np.allclose(qd1.Λ,8.617018973053206), qd1.Λ
        # # qd2 = qpm(1550,810,width=3,depth=5,ape=1.0,rpe=0.9,sell='fan',Type='zzz',res=0.1) # print(qd2.Λ)
        # qd3 = Ktpwaveguide(3,5,1,0.9,0,'fan','chestep',step=0.1).qpm(1550,810) # print(qd3.Λ,qd3.Λ-qd2.Λ)
        # assert np.allclose(qd3.Λ,8.546358532830164), qd3.Λ
        # # qd4 = qpm(1550,810,width=3,depth=5,ape=1.0,rpe=0.9,sell='ktpmik',Type='zzz',res=0.1) # print(qd4.Λ)
        # qd5 = Ktpwaveguide(3,5,1,0.9,0,'ktp','mikstep',step=0.1).qpm(1550,810) # print(qd5.Λ,qd5.Λ-qd4.Λ)
        # assert np.allclose(qd5.Λ,8.690793594695558),qd5.Λ
        # # qd6 = qpm(1550,810,width=3,depth=5,ape=1.0,rpe=0.9,sell='ktp',Type='zzz',res=0.05) # print(qd6.Λ) # 8.61164521502637
        # qd7 = Ktpwaveguide(w=3,d=5,c=1,p=0.9,step=0.05).qpm(1550,810) # print(qd7.Λ,qd7.Λ-qd6.Λ) # 8.613879085732405 0.0022338707162514737
        # assert np.allclose(qd7.Λ,8.613879085732405), qd7.Λ

def ktpstripwgtest(atol=1e-3):
    mdv = Ktpstripwaveguide(λ=1064,pol='v',step=(0.2,0.05),sy=0.4).modesolve() # mdv.indexplot() # mdv.indexprofiley().plot() # mdv.plot()
    assert np.allclose(mdv.dneff,0.01682794123594067,atol=atol), mdv.dneff
    mdv = Ktpstripwaveguide(λ=1064,pol='v',step=(0.2,0.05),sx=2,sy=0.4).modesolve()
    assert np.allclose(mdv.dneff,0.01010249661399154,atol=atol), mdv.dneff
    mdh = Ktpstripwaveguide(λ=1064,pol='h',step=(0.2,0.05),sx=2,sy=0.4).modesolve()
    assert np.allclose(mdh.dneff,0.05048324425084494,atol=atol), mdh.dneff
    assert np.allclose(mdh.mfdx,1.5635765389331273,atol=atol), mdh.mfdx
    assert np.allclose(mdh.mfdy,0.944443978994313,atol=atol), mdh.mfdy
    # mdh = Ktpstripwaveguide(λ=1064,pol='h',step=0.20,sx=2,sy=0.4).modesolve(); print(f"{mdh.dneff:g},  {mdh.mfdx:g},  {mdh.mfdy:g}")
    # mdh = Ktpstripwaveguide(λ=1064,pol='h',step=(0.20,0.05),sx=2,sy=0.4).modesolve(); print(f"{mdh.dneff:g},  {mdh.mfdx:g},  {mdh.mfdy:g}")
    # mdh = Ktpstripwaveguide(λ=1064,pol='h',step=(0.05,0.05),sx=2,sy=0.4).modesolve(); print(f"{mdh.dneff:g},  {mdh.mfdx:g},  {mdh.mfdy:g}")
def sinwaveguidetest(atol=1e-3):
    dd,dm = dict(w=1,wfn=3,step=(0.2,0.02),split=2),dict(method='exact',mode=0,nummodes=9,boundary='0000',verbose=0)
    mdv = Sinwaveguide(λ=1064,pol='v',**dd).modesolve(**dm)
    mdh = Sinwaveguide(λ=1064,pol='h',**dd).modesolve(**dm)
    # mdh.plotindex(vmin=mdh.nsub-0.2,contour=0)
    # Wave.plot(mdh.indexprofilex(),mdv.indexprofilex(),l='01')
    # print('mdv.dneff',mdv.dneff,'mdv.mfdx',mdv.mfdx,'mdv.mfdy',mdv.mfdy)
    # print('mdh.dneff',mdh.dneff,'mdh.mfdx',mdh.mfdx,'mdh.mfdy',mdh.mfdy)
    # mfdyv = sum([Sinwaveguide(λ=1064,pol='v',**dd).modesolve(**dm).mfdy for _ in range(10)])/10; print('mdv.mfdy',mdv.mfdy,'mfdyv',mfdyv)
    # mfdyh = sum([Sinwaveguide(λ=1064,pol='h',**dd).modesolve(**dm).mfdy for _ in range(10)])/10; print('mdh.mfdy',mdh.mfdy,'mfdyh',mfdyh)
    assert np.allclose(mdv.neff,1.6652752077639206,atol=atol), mdv.neff
    assert np.allclose(mdv.mfdx,3.9923032911225795,atol=atol), mdv.mfdx
    assert np.allclose(mdv.mfdy,0.7642899101753956,atol=atol), mdv.mfdy
    assert np.allclose(mdh.neff,1.7542410133610091,atol=atol), mdh.neff
    assert np.allclose(mdh.mfdx,3.9824564832960476,atol=atol), mdh.mfdx
    assert np.allclose(mdh.mfdy,0.6138269287741712,atol=atol), mdh.mfdy
    # Sinwaveguide(λ=1064,pol='h',step=(0.1,0.01)).modesolve(method='exact',mode=0,nummodes=9,boundary='0000',verbose=True)
def bendlosstest():
    def sinmd(λ,wsn,wfn=None,fsn=0,ffn=1,pol='v',nm=9,mode=0,allmodes=True,boundary=None,method='exact',step=(0.1,0.02),bounds=(-10,10,-10,2)):
        boundary = boundary if boundary is not None else 'SSSS' if 'v'==pol else 'AAAA'
        wg = Sinwaveguide(λ=λ,w=wsn,wfn=wfn,bf=fsn,bffn=ffn,pol=pol,step=step,bounds=bounds)
        if allmodes:
            return wg(pol=None).solve(boundary=boundary,method=method,mode=mode,nummodes=nm)
        return wg.solve(boundary=boundary,method=method,mode=mode,nummodes=nm)
        # return wg.newmodesolve(boundary=boundary,method=method,mode=mode,nummodes=nm)
        # return wg.oldmodesolve(boundary=boundary,method=method,mode=mode,nummodes=nm,allmodes=allmodes)
    mdv = sinmd(1550,2,pol='v')[0] # mdv.plot(xlim=(-2,2),ylim=(-2,1))
    mdh = sinmd(1550,2,pol='h')[0]
    print(f'd4σx v {mdv.d4sigmax():g} µm')
    print(f'd4σx h {mdh.d4sigmax():g} µm')
    from wavedata import Wave
    C = mdv.weaklyguidinglossslope(); print(f'Cv {C:g}mm⁻¹') # Rsafe {4e3/C:g}µm')
    C = mdh.weaklyguidinglossslope(); print(f'Ch {C:g}mm⁻¹') # Rsafe {4e3/C:g}µm')
    # mdv.bendloss().plot(log=1,x='R (mm)',y='α (mm⁻¹)',grid=1)
    # mdh.bendloss().plot(log=1,x='R (mm)',y='α (mm⁻¹)',grid=1)
    wg = Ktpwaveguide(λ=1550,w=4,d=5,c=1,p=0.9,r=0,crystal='ktp',crystalstep='ktpsurf',sellcover='air',pol='v',cut='zyx',bounds=(-20,20,-30,2),step=0.2)
    # md = wg.newmodesolve(method='exact',mode=0,nummodes=49,boundary=None,verbose=True) # boundary = 'SSSS' if 'v'==pol else 'AAAA'
    md = wg.solve(method='exact',mode=0,nummodes=49,boundary=None,verbose=True) # boundary = 'SSSS' if 'v'==pol else 'AAAA'
    C = md.weaklyguidinglossslope(); print(f'Cwg {C:g}mm⁻¹') # Rsafe {4e3/C:g}µm')
    rs = wrange(0,10,0.01)
    ls = Wave(np.exp(2-C*rs),-rs)
    C = md.lossslope(show=0)
    print(f'C {C:g}mm⁻¹')
    md.bendloss().plot(ls,log=1,x='R (mm)',y='α (mm⁻¹)',grid=1)
def ktpbendlossvsroccompare():
    # compare loss predictions to data from igor beamprop simulations
    from wavedata import Wave,deal
    def ktpmode(λ=1064,w=4,d=10,c=1,boundary='SSSS'):
        wg = Ktpwaveguide(λ=λ,w=w,d=d,c=c,p=1,r=0,crystal='ktp',crystalstep='ktpsurf',sellcover='air',pol='v',cut='zyx',bounds=(-20,20,-30,2),step=0.2)
        # md = wg.newmodesolve(method='exact',mode=0,nummodes=19,boundary=boundary,verbose=True) # boundary = 'SSSS' if 'v'==pol else 'AAAA'
        md = wg.solve(method='exact',mode=0,nummodes=19,boundary=boundary,verbose=True)
        return md
    def slope(md,dbpermm=False):
        C = md.weaklyguidinglossslope()
        rs = wrange(0,2,0.01)
        w = Wave(np.exp(4-C*rs),rs)
        return w if not dbpermm else 20*np.log10(np.exp(1))*w
    # md0,md1 = ktpmode(boundary='SSSS'),ktpmode(boundary='AAAA')
    # md0,md1 = ktpmode(w=4),ktpmode(w=2); Wave.plot(md0.bendloss(dbpermm=1).rename(4),md1.bendloss(dbpermm=1).rename(2),slope(md0,1).rename(4),slope(md1,1).rename(2),c='0101',l='13',log=1,x='R (mm)',y='loss (dB/mm)',grid=1)
    # md0,md1 = ktpmode(c=1),ktpmode(c=0.5)
    # Wave.plot(md0.bendloss(),md1.bendloss(),slope(md0),slope(md1),c='0101',l='13',log=1,x='R (mm)',y='α (mm⁻¹)',grid=1)
    # Wave.plot(md0.bendloss(dbpermm=1),md1.bendloss(dbpermm=1),slope(md0,1),slope(md1,1),c='0101',l='13',log=1,x='R (mm)',y='loss (dB/mm)',grid=1)
    md0,md1 = ktpmode(c=1),ktpmode(c=0.5)
    w4d10full = Wave.fromxsandys(*deal(2,[100,309.50452,200,112.61481,300,44.118671,400,19.758842,500,8.5540047,600,3.8217947,700,1.5797693,800,0.69431496,900,0.30092248,1000,0.11991208]))
    w4d10half = Wave.fromxsandys(*deal(2,[400,112.96673,600,60.663437,800,36.742752,1000,20.564323,1200,11.947983,1400,7.3753576,1600,4.7791028,1800,3.1949158,2000,2.1472785]))
    w4d10full,w4d10half = w4d10full.scalex(1e-3).rename('100% Rb Beamprop').setplot(m='o'),w4d10half.scalex(1e-3).rename('50% Rb Beamprop').setplot(m='o')
    Wave.plot(w4d10full,md0.bendloss(dbpermm=1).rename('100% Rb Marcatili69'),slope(md0,1).rename('100% Rb Vlasov04'),w4d10half,md1.bendloss(dbpermm=1).rename('50% Rb Marcatili69'),slope(md1,1).rename('50% Rb Vlasov04'),
        x='radius of curvature (mm)',y='loss (dB/mm)',xlim=(0,3),ylim=(1e-3,1e3),c='000111',l='013013',grid=1,log=1,fontsize=8,save='ktp bendloss vs roc,rb')
    x2,y2,x4,y4,x8,y8 = deal(6,[100,391.3624,100,309.50452,100,nan,200,120.78433,200,112.61481,200,110.53934,300,55.320496,300,44.118671,300,44.000259,400,29.713919,400,19.758842,400,18.493111,500,17.316256,500,8.5540047,500,8.0642719,600,9.7940741,600,3.8217947,600,3.5370941,700,5.9637418,700,1.5797693,700,1.350485,800,3.6070435,800,0.69431496,800,nan,900,2.1077945,900,0.30092248,900,nan,1000,1.2704755,1000,0.11991208,1000,nan])
    a2,b2,a4,b4,a8,b8 = deal(6,[200,248.21179,200,221.47438,200,nan,400,126.75099,400,108.39738,400,110.26835,600,76.863457,600,60.918232,600,57.674107,800,54.105507,800,36.131931,800,33.653828,1000,38.465645,1000,23.074718,1000,21.869257,1200,31.168022,1200,14.294477,1200,13.329087,1400,24.135353,1400,9.6352568,1400,7.8211188,1600,18.601017,1600,6.7952871,1600,4.6540327,1800,14.981643,1800,4.7407517,1800,2.8664806,2000,12.553557,2000,3.2145658,2000,1.8298293])
    u2,u4,u8 = Wave(y2,x2,'2µm 100% BPM').scalex(1e-3).setplot(m='o'),Wave(y4,x4,'4µm 100% BPM').scalex(1e-3).setplot(m='o'),Wave(y8,x8,'8µm 100% BPM').scalex(1e-3).setplot(m='o')
    v2,v4,v8 = Wave(b2,a2,'2µm 50% BPM').scalex(1e-3).setplot(m='o'),Wave(b4,a4,'4µm 50% BPM').scalex(1e-3).setplot(m='o'),Wave(b8,a8,'8µm 50% BPM').scalex(1e-3).setplot(m='o')
    uu2,uu4,uu8 = ktpmode(w=2,c=1.0).bendloss(1).rename('2µm 100% FD'),ktpmode(w=4,c=1.0).bendloss(1).rename('4µm 100% FD'),ktpmode(w=8,c=1.0).bendloss(1).rename('8µm 100% FD')
    vv2,vv4,vv8 = ktpmode(w=2,c=0.5).bendloss(1).rename('2µm 50% FD'),ktpmode(w=4,c=0.5).bendloss(1).rename('4µm 50% FD'),ktpmode(w=8,c=0.5).bendloss(1).rename('8µm 50% FD')
    Wave.plot(u2,u4,u8,v2,v4,v8,uu2,uu4,uu8,vv2,vv4,vv8,c='012012012012',l='000111333444',seed=5,
        x='radius of curvature (mm)',y='loss (dB/mm)',xlim=(0,3),ylim=(1e-3,1e3),grid=1,log=1,fontsize=8,save='ktp bendloss vs roc,rb,width')
def ktpstriploss():
    args = dict(w=4,d=5,c=1,p=0.9,r=0,sx=None,sy=0.4,crystal='ktp',crystalstep='ktpsurf',sellcover='air',sellstrip='sin',pol='v',cut='zyx',bounds=None,step=(0.2,0.05))
    mdv = Ktpstripwaveguide(λ=1064,**args).modesolve() # mdv.indexplot() # mdv.indexprofiley().plot() # mdv.plot()
def planewavetest(λ=1550,pol='h',plot=0):
    wg = Planewaveguide(λ=λ,pol=pol,sell=None,bounds=None,step=0.1)
    md = wg.modesolve(boundary=None)
    if plot:
        # Wave.plot(md.ex,md.ey,seed=7,log=0,grid=1,c='44',l='23',ylim=(0,1.1),xlim='f',x='µm',y='relative field',save='planewave mode comparison')
        md0 = wg.modesolve(boundary='0000',method='exact',mode=0,nummodes=19)
        md1 = wg.modesolve(boundary='SSSS',method='exact',mode=0,nummodes=19)
        md2 = wg.modesolve(boundary='AAAA',method='exact',mode=0,nummodes=19)
        ws = [md0.ex.rename('0000'),md1.ex.rename('SSSS'),md2.ex.rename('AAAA'),md0.ey.rename('0000'),md1.ey.rename('SSSS'),md2.ey.rename('AAAA')]
        Wave.plot(*ws,seed=7,log=0,grid=1,c='012',l='222333',ylim=(0,None),xlim='f',x='µm',y='relative field',save='planewave mode comparison')
    # assert np.allclose(md.dneff,0) and np.allclose(md.neff,1) and 5e4<md.mfdx and 4e4<md.mfdy, f'dneff {md.dneff:g}, neff {md.neff:g}, mfdx {md.mfdx:g}, mfdy {md.mfdy:g}'
    wg = Planewaveguide(λ=λ,pol=pol,sell=None,bounds=None,step=0.1)
    # wg = Planewaveguide(λ=λ,pol=pol,n=1+1j*1e-9,sell=None,bounds=None,step=0.1)
    md = wg.modesolve(boundary='neumann')
    assert np.allclose(md.dneff,0) and np.allclose(md.neff,1) and 5e4<md.mfdx and 4e4<md.mfdy, f'dneff {md.dneff:g}, neff {md.neff:g}, mfdx {md.mfdx:g}, mfdy {md.mfdy:g}'
def modetimetest(): # slightly sub-linear
    from time import time
    def mdtime(i):
        t0 = time()
        md = Ktpwaveguide(λ=1060,pol=None,step=0.5).solve(method='exact',verbose=0,nummodes=i)
        return time()-t0
    ns = sorted(list({int(1.3**i) for i in range(99) if 1.3**i<=500})); print(ns)
    ts = [mdtime(n) for n in track(ns)]
    w = Wave(ns,ts,m='o')
    u,cc = w.linefit(coef=2,b=0)
    Wave.plot(w,u,x='time (s)',y='modes')
    print(cc,f'{sum(ts)}s total time')
def fibertest(extended=False,plot=False):
    if 1: # 5s
        wg = Stepfiberwaveguide(λ=1550,r=0.5*8.5,ncore=1.449,nclad=1.444,pol='v',sell=None,bounds=None,step=0.2) # Corning PM1550 fiber # wg.plot()
        md0 = wg.modesolve(boundary='SSSS') # md.plot()
        md1 = wg.modesolve(boundary='0000') # md.plot()
        fd = wg.fibermode() # fd.plot()
        w0 = md0.ex.normalize().rename('modesolver 1')
        w1 = md1.ey.normalize().rename('modesolver 2')
        u = fd.ex.normalize().rename('analytic')
        if plot: Wave.plot(w0,w1,u,log=1,grid=1,l='123',x='µm',y='relative field',save='fiber mode comparison',seed=2)
        print(f'Δn {md0.dneff:g} Δnfd {fd.dneff:g}')
        print('overlap',md0.overlap(fd))
    if extended: # 60s
        mdv = wg(pol='v').solve(method=None,mode=0,nummodes=9,boundary='0000',verbose=True)
        mdh = wg(pol='h').solve(method=None,mode=0,nummodes=9,boundary='0000',verbose=True)
        print(mdv.dneffs[-3:])
        print(mdh.dneffs[-3:])
        if plot:
            for i in (0,1,2): mdv[i].plot(legendtext=f'V mode {i}\nΔn = {mdv[i].dneff:+.4f}',aspect=1,save=f'fiber test mode {i} v')
            for i in (0,1,2): mdh[i].plot(legendtext=f'H mode {i}\nΔn = {mdh[i].dneff:+.4f}',aspect=1,save=f'fiber test mode {i} h')
        wg = Stepfiberwaveguide(λ=1500,r=3,ncore=1.45,nclad=1,pol='h',sell=None,bounds=5,step=0.2)
        for step in (0.1,0.05,0.02):#,0.01,0.005): # 0.0025 memory error
            print(f"{step:.3f} neff {wg(step=step).modesolve(boundary='SSSS',method='exact',mode=0,nummodes=2,verbose=0).neff:.8f}")
def tflnridgetest(plot=False):
    if 1: # mgln ridge sellmeier comparison
        wg = Ridgewaveguide(w=10,h=10,etch=10,cut='zyx',crystal='mgln',sellbase='sio2',sellcover='air')
        λs = [400,600,800,1000,1200,1400,1600]
        from sellmeier import index
        u0 = W(index(λs,'mglnridgewgz')-0*index(λs,'mglnz'),λs,c='k',l='3')
        u1 = W([wg(λ=λ,pol='v').modesolve().neff-0*index(λ,'mglnz') for λ in λs],λs)
        v0 = W(index(λs,'mglnridgewgy')-0*index(λs,'mglny'),λs,c='k',l='2')
        v1 = W([wg(λ=λ,pol='h').modesolve().neff-0*index(λ,'mglny') for λ in λs],λs)
        if plot:
            W.plot(u1,u0,v1,v0,x='λ (nm)',y='$n_{eff}$',save='mglnridge neff vs λ')
    if 1:
        wg = Ridgewaveguide(λ=1550,w=2,h=1,etch=0.5,cut='xzy',crystal='ln',sellbase='sio2',sellcover='air',bounds=2,step=0.1)
        mdh = wg(pol='h').modesolve(); print(mdh.neff); # mdh.plot()
        mdh = wg(pol='h',bounds=3).modesolve(); print(mdh.neff)
        mdv = wg(pol='v').modesolve(); print(mdv.neff); # mdv.plot()
        mdv = wg(pol='v',bounds=3).modesolve(); print(mdv.neff)
    print('~')
    wg = Ridgewaveguide(λ=1550,w=2,h=1,etch=0.5,cut='xzy',crystal='ln',sellbase='sio2',sellcover='air',bounds=2,step=0.1)
    mdh = wg(pol='h',split=0).modesolve(); print(mdh.neff); # mdh.plot()
    mdh = wg(pol='h',split=3).modesolve(mode=1); print(mdh[0].neff); print(mdh[1].neff); print(mdh.couplinglength(),'mm') # mdh.plot()
    mdv = wg(pol='v',split=0).modesolve(); print(mdv.neff); # mdv.plot()
    mdv = wg(pol='v',split=3).modesolve(mode=1); print(mdv[0].neff); print(mdv[1].neff); print(mdv.couplinglength(),'mm') # mdv.plot()
    for pol in 'hv':
        print(pol.upper())
        md3 = wg(pol=pol,split=3).modesolve(mode=1)
        md4 = wg(pol=pol,split=4).modesolve(mode=1)
        print('Lc',md3.Lc,md4.Lc,'split',md3.args.split,md4.args.split,'Lr (mm)',md3.Lr(1,md=md4))
    def Lcvssplit(pol='h',width=2):
        splits = wrange(width-1,width+2,0.1)
        splits = wrange(width-1,width,0.1,aslist=1) + wrange(width+0.05,width+0.5,0.05,aslist=1) + wrange(width+0.5+0.1,width+2,0.1,aslist=1)
        wg = Ridgewaveguide(λ=1550,w=width,h=1,etch=0.5,cut='xzy',crystal='ln',sellbase='sio2',sellcover='air',bounds=2,step=0.05)
        Lcs = [wg(pol=pol,split=split).modesolve(mode=1).couplinglength(symmetrywarning=0) for split in splits]
        n = len([s for s in splits if width<=s])
        u = Wave(Lcs,splits,l='3',mf='w')
        v = Wave(Lcs[-n:],splits[-n:],pol.upper(),l='0')
        return u,v
    (hu,hv),(vu,vv) = Lcvssplit('h'),Lcvssplit('v')
    if plot: Wave.plot(hu,hv,vu,vv,c='0011',clip=0,m='o',ms=3,log=1,grid=1,xlim='f',x='split (µm)',y='coupling length (mm)',save='ln ridge coupling length vs split')
    if 0:
        wg = Ridgewaveguide(λ=1550,w=2,h=1,etch=0.5,cut='xzy',crystal='ln',sellbase='sio2',sellcover='air',bounds=2,step=0.1)
        W.plot(wg(pol='h').modesolve().bendloss().rename('H'),wg(pol='v').modesolve().bendloss().rename('V'),xlim='f',log=1,x='R (mm)',y='bend loss (dB/mm)',grid=1,save=f'ln ridge bend loss vs split')
    if 1:
        wg0 = Ridgewaveguide(λ=1550,w=2,h=1,etch=0.5,cut='xzy',crystal='ln',sellbase='sio2',sellcover='air',bounds=2,step=0.1); wg0.nn.plot(show=plot)
        wg1 = Trapezoidalridgewaveguide(λ=1550,w=2,h=1,etch=0.5,cut='xzy',crystal='ln',sellbase='sio2',sellcover='air',bounds=2,step=0.1); wg1.nn.plot(show=plot)
        print('(wg0.nn-wg1.nn).max()',(wg0.nn-wg1.nn).max())
        print('(wg0.nn-wg1.nn).min()',(wg0.nn-wg1.nn).min())
    if plot:
        Trapezoidalridgewaveguide(λ=1550,w=2,h=1,etch=0.5,cut='xzy',crystal='ln',sellbase='sio2',sellcover='air',bounds=2,step=0.1,trapezoidangle=0.5).plot()
        Trapezoidalridgewaveguide(λ=1550,w=2,h=1,etch=0.5,cut='xzy',crystal='ln',sellbase='sio2',sellcover='air',bounds=2,step=0.1,trapezoidangle=pi/4,split=5).plot()
        Trapezoidalridgewaveguide(λ=1550,w=2,h=1,etch=0.5,cut='xzy',crystal='ln',sellbase='sio2',sellcover='air',bounds=2,step=0.1,trapezoidangle=0.4*pi,split=3).plot()
        Trapezoidalridgewaveguide(λ=1550,w=2,h=1,etch=0.5,cut='xzy',crystal='ln',sellbase='sio2',sellcover='air',bounds=2,step=0.1,trapezoidangle=-pi/4,split=5).plot()
def zhutest():
    wg = Ridgewaveguide(10,10,5,step=0.5,λ=1064,pol='v')
    md = wg(pol='v').solve(solver='zhu',boundary=None,verbose=0); assert np.allclose(md.neff,2.145788301078139,rtol=1e-4), md.neff
    md = wg(pol='h').solve(solver='zhu'); assert np.allclose(md.neff,2.2276504639690673,rtol=1e-3), md.neff
    wg = Ktpwaveguide(λ=1064)
    # md = wg(pol='v').modesolve(); assert np.allclose(md.neff,1.8376751933545603,rtol=1e-4), md.neff
    # md = wg(pol='h').modesolve(); assert np.allclose(md.neff,1.75834719810613,rtol=1e-4), md.neff
    md = wg(pol='v').solve(solver='zhu'); assert np.allclose(md.neff,1.8378810540904522,rtol=1e-4), md.neff
    md = wg(pol='h').solve(solver='zhu'); assert np.allclose(md.neff,1.758242723577965,rtol=1e-4), md.neff
def zhuinfo(plot=False):
    wg = Ktpwaveguide(λ=1064,step=0.5,pol='v')
    md = wg(pol='v').solve(solver='zhu',boundary=None,verbose=0); print(md.neff)
    # print all mode info that we possible can
    print('neff',md.neff)
    print('dneff',md.dneff)
    print('mfdx',md.mfdx,'µm')
    print('mfdy',md.mfdy,'µm')
    print('fibercouplingefficiency',md.fibercoupling())
    print('mode area',md.modearea(),'µm²')
    print('modeid',md.modeid())
    if plot:
        md.plot()
        md.plotindex()
        md.indexprofiley().plot(x='depth (µm)',y='index',xlim='f',ylim=(1.8,None))

if __name__ == '__main__':
    if 0:
        simplesttest()
        wgtest('v','zyx')
        wgtest('h','xzy')
        # ktpwgtest()
        # ktpwgtest(extended=1)
        ridgeshgtest(1)
        ktpstriploss()
        planewavetest(plot=0)
        fibertest()
        fibertest(extended=1)
        sinwaveguidetest()
        tflnridgetest()
        modetimetest() # 211s
        bendlosstest() # 260s
        ktpbendlossvsroccompare() #394s
        ridgewgtest()
        # ktpstripwgtest() # h,v Δn swapped?
    zhutest()
    zhuinfo(1)

    # from modes import newmodesolver,modesolver
    # from sellmeier import index
    # md = newmodesolver(1064,10,10,5,0,'mglnridgez',res=0.5,method=None); assert np.allclose(md.neff,2.1456840136387503,rtol=1e-4), md.dneff
    # md = newmodesolver(1064,10,10,5,0,'mglnridgey',res=0.5,method=None); print(md.dneff,md.neff)
    # md = newmodesolver(1064,10,10,5,0,'mglnridgez',res=0.5,verbose=0,method='zhu'); assert np.allclose(md.neff,2.1456840136387503,rtol=1e-4), md.dneff; md.plot()
    # md = newmodesolver(1064,6,10,5,0,'mglnridgez',res=0.5,verbose=0,method=None); print(md.dneff,md.neff); md.plot()
    # md = newmodesolver(1064,6,10,5,0,'mglnridgez',res=0.5,verbose=0,method='zhu'); print(md.dneff,md.neff); md.plot()
    # md = newmodesolver(1064,9,11,11,0,'mglnridgey',res=0.5,method=None); print(index(1064,'mglnridgewg'),md.neff); md.plot()
    # md = newmodesolver(1064,9,11,11,0,'mglnridgey',res=0.5,verbose=0,method='zhu'); print(index(1064,'mglnridgewg'),md.neff); md.plot()

    # print(md.dneff,md.neff)
    # md = newmodesolver(1064,10,10,5,0,'mglnridgey',res=0.5,verbose=0,method='zhu'); assert np.allclose(md.neff,2.1456325746604055,rtol=1e-4), md.dneff # should be 2.23
    # print(index(1064,'mglnz'),index(1064,'mglny'))
    # wg = Ridgewaveguide(10,10,5,step=0.5,λ=1064,pol='v',crystal='mgln')
    # md = wg(pol='v').modesolve(verbose=0); assert np.allclose(md.neff,2.1456931244338113,rtol=1e-4), md.dneff
    # md = wg(pol='h').modesolve(verbose=0); assert np.allclose(md.neff,2.2275608772173148,rtol=1e-4), md.dneff
    # md = wg(pol='v').modesolve(nummodes=2,method='isotropic',verbose=0); assert np.allclose(md.dneff,0.6959977153270969,rtol=1e-4), md.dneff
    # wg = Ridgewaveguide(6,10,5,step=0.5,λ=1064,pol='v',crystal='mgln')
    # md = wg(pol='v').solve(solver='zhu',boundary='dirichlet',verbose=0); print(md.dneff,md.neff); md.plot()

    # from modes import modesolver,newmodesolver
    # md = modesolver(1064,8,1.85,18,7.25,sell='lnz',res=0.5,mode=0,nummodes=2,verbose=False,method='isotropic',cachelookup=0); print(md.dneff)
    # md = newmodesolver(1064,8,1.85,18,7.25,sell='lnz',res=0.5,mode=0,nummodes=2,verbose=False,method='isotropic',cachelookup=0); print(md.dneff)
    # md = newmodesolver(1064,8,1.85,18,7.25,sell='lnz',res=0.5,mode=0,nummodes=2,verbose=False,method='zhu',cachelookup=0); print(md.dneff) # assert np.allclose(0.012255,md.dneff,rtol=1e-3)
    # 0.012254700305563304
    # 0.01225470030263276
    # 0.012795287354452789

