import numpy as np
from waveguide import Waveguide,Planewaveguide,Stepfiberwaveguide
from waveguide import Corningpmfiberwaveguide,Ktpwaveguide,Ridgewaveguide,Rpewaveguide
from waveguide import Qpmdata,Trapezoidalridgewaveguide,Ktpstripwaveguide,Sinwaveguide
from wavedata import Wave,Wave2D,wrange,timeit,track
from numpy import pi,nan
W = Wave

def corningpmfiberexample(λ=1064,fiber=980,step=0.2,plot=False):
    # λ = wavelength in nm
    # fiber = corning fiber number, e.g. 980 for PM980
    # step = grid step size in µm, small step size is more accurate but takes longer
    wg = Corningpmfiberwaveguide(λ=λ,fiber=fiber,step=step)
    md = wg.solve(solver='zhu',mode=0,nummodes=99)
    print(f'{md.neff.real:g} effective index')
    print(f'{md.mfdx:g}µm mode field diameter width')
    print(f'{md.mfdy:g}µm mode field diameter height')
    print(f'{md.modearea():g} µm² mode area')
    print(f'{md.modecount()} modes, {md.guidedmodecount()} guided modes')
    if plot:
        s = f'corning pm fiber waveguide mode{md.modenum} at {λ}nm'
        md.plot(x='width (µm)',y='depth (µm)',save=f'mode profile, {s}')
        md.indexprofilex().plot(x='x (µm)',y='index',grid=1,xlim='f',save=f'index profile x, {s}')
        md.indexprofiley().plot(x='y (µm)',y='index',grid=1,xlim='f',save=f'index profile y, {s}')
        md.angulardistributionx().abs().plot(x='angle (degrees)',y='field',grid=1,xlim='f',save=f'angular distribution x, {s}')
        md.angulardistributiony().abs().plot(x='angle (degrees)',y='field',grid=1,xlim='f',save=f'angular distribution y, {s}')
        md.ex.abs().plot(x='x (µm)',y='field',grid=1,xlim='f',save=f'field distribution x, {s}')
        md.ey.abs().plot(x='y (µm)',y='field',grid=1,xlim='f',save=f'field distribution y, {s}')
def ktpwaveguideexample(λ=1064,width=4,depth=5,conc=1,anneal=1,reverseexchange=0,step=0.1,pol='v',plot=False):
    # λ = wavelength in nm
    # width = waveguide litho mask width in µm
    # depth = waveguide fwhm depth in µm
    # conc = fractional rubidium ion surface concentration after ion exchange
    # anneal = relative surface concentration after post-exchange annealing, anneal=1 is no anneal by default
    # reverseexchange = reverse exchange depth
    # step = grid step size in µm
    # pol = 'v' or 'h' for vertical or horizontal polarization
    wg = Ktpwaveguide(λ=λ,w=width,d=depth,c=conc,p=anneal,r=reverseexchange,step=step,pol=pol)
    md = wg.solve(solver='zhu',mode=0,nummodes=99)
    print(f'{md.neff.real:g} effective index')
    print(f'{md.mfdx:g}µm mode field diameter width')
    print(f'{md.mfdy:g}µm mode field diameter height')
    print(f'{md.modearea():g} µm² mode area')
    print(f'{md.modecount()} modes, {md.guidedmodecount()} guided modes')
    # print(f'{md.modeid()} mode id')
    # print(md.fiberoverlap(fiber=980))
    # print(f'{md.fibercoupling():g} fiber coupling efficiency to PM{md.fiber()} fiber')
    if plot:
        s = f'ktp waveguide {md.pol.upper()} mode{md.modenum} at {λ}nm'
        md.plot(x='width (µm)',y='depth (µm)',save=f'mode profile, {s}')
        md.indexprofilex().plot(x='x (µm)',y='index',grid=1,xlim='f',save=f'index profile x, {s}')
        md.indexprofiley().plot(x='y (µm)',y='index',grid=1,xlim='f',save=f'index profile y, {s}')
        md.angulardistributionx().plot(x='angle (degrees)',y='field',grid=1,xlim='f',save=f'angular distribution x, {s}')
        md.angulardistributiony().plot(x='angle (degrees)',y='field',grid=1,xlim='f',save=f'angular distribution y, {s}')
        md.ex.abs().plot(x='x (µm)',y='field',grid=1,xlim='f',save=f'field distribution x, {s}')
        md.ey.abs().plot(x='y (µm)',y='field',grid=1,xlim='f',save=f'field distribution y, {s}')
def ridgewaveguideexample(λ=1550,width=10,depth=10,etchdepth=5,step=0.2,plot=False):
    wg = Ridgewaveguide(w=width,h=depth,etch=etchdepth,bf=1,split=0,crystal='mgln',sellbase='sio2',sellcover='air',pol='v',cut='zyx',λ=λ,bounds=None,step=step)
    md = wg.solve(solver='zhu',mode=0,nummodes=99)
    print(f'{md.neff.real:g} effective index')
    print(f'{md.mfdx:g}µm mode field diameter width')
    print(f'{md.mfdy:g}µm mode field diameter height')
    print(f'{md.modearea():g} µm² mode area')
    print(f'{md.modecount()} modes, {md.guidedmodecount()} guided modes')
    if plot:
        s = f'mgln ridge waveguide mode{md.modenum} at {λ}nm'
        md.plot(x='width (µm)',y='depth (µm)',save=f'mode profile, {s}')
        md.indexprofilex().plot(x='x (µm)',y='index',grid=1,xlim='f',save=f'index profile x, {s}')
        md.indexprofiley().plot(x='y (µm)',y='index',grid=1,xlim='f',save=f'index profile y, {s}')
        md.angulardistributionx().abs().plot(x='angle (degrees)',y='field',grid=1,xlim='f',save=f'angular distribution x, {s}')
        md.angulardistributiony().abs().plot(x='angle (degrees)',y='field',grid=1,xlim='f',save=f'angular distribution y, {s}')
        md.ex.abs().plot(x='x (µm)',y='field',grid=1,xlim='f',save=f'field distribution x, {s}')
        md.ey.abs().plot(x='y (µm)',y='field',grid=1,xlim='f',save=f'field distribution y, {s}')
def rpewaveguideexample(λ=1550,width=8,sadepth=1.9,annealtime=23.5,reversetime=24.5,annealtemp=320,reversetemp=300,step=0.2,plot=False):
    # λ = wavelength in nm
    # width = waveguide litho mask width in µm
    # sadepth = soft anneal depth in µm
    # annealtime = hard anneal time in hours
    # reversetime = reverse exchange time in hours
    # annealtemp = hard anneal temperature in °C
    # reversetemp = reverse exchange temperature in °C
    wg = Rpewaveguide(w=width,sa=sadepth,a=annealtime,r=reversetime,a2=0,at=annealtemp,rt=reversetemp,a2t=None,split=0,bf=1,crystal='ln',crystalstep='lnzcutstep',sellcover='air',pol='v',cut='zyx',λ=λ,bounds=None,step=step,diffres=0.1)
    md = wg.solve(solver='zhu',mode=0,nummodes=99)
    print(f'{md.neff.real:g} effective index')
    print(f'{md.mfdx:g}µm mode field diameter width')
    print(f'{md.mfdy:g}µm mode field diameter height')
    print(f'{md.modearea():g} µm² mode area')
    print(f'{md.modecount()} modes, {md.guidedmodecount()} guided modes')
    # print(f'{md.apeprotondose():g} ape proton dose')
    # print(f'{md.protondose():g} proton dose')
    if plot:
        s = f'ln rpe waveguide mode{md.modenum} at {λ}nm'
        md.plot(x='width (µm)',y='depth (µm)',save=f'mode profile, {s}')
        md.indexprofilex().plot(x='x (µm)',y='index',grid=1,xlim='f',save=f'index profile x, {s}')
        md.indexprofiley().plot(x='y (µm)',y='index',grid=1,xlim='f',save=f'index profile y, {s}')
        md.angulardistributionx().abs().plot(x='angle (degrees)',y='field',grid=1,xlim='f',save=f'angular distribution x, {s}')
        md.angulardistributiony().abs().plot(x='angle (degrees)',y='field',grid=1,xlim='f',save=f'angular distribution y, {s}')
        md.ex.abs().plot(x='x (µm)',y='field',grid=1,xlim='f',save=f'field distribution x, {s}')
        md.ey.abs().plot(x='y (µm)',y='field',grid=1,xlim='f',save=f'field distribution y, {s}')
def rpedirectionalcoupler(λ=1550,split=12,width=8,sadepth=1.9,annealtime=23.5,reversetime=24.5,annealtemp=320,reversetemp=300,step=0.2,plot=False):
    # split = directional coupler center-to-center waveguide separation in µm
    wg = Rpewaveguide(w=width,sa=sadepth,a=annealtime,r=reversetime,a2=0,at=annealtemp,rt=reversetemp,a2t=None,split=split,bf=1,crystal='ln',crystalstep='lnzcutstep',sellcover='air',pol='v',cut='zyx',λ=λ,bounds=None,step=step,diffres=0.1)
    md = wg.solve(solver='zhu',mode=1,nummodes=2)
    print(f'{md.modecount()} modes, {md.guidedmodecount()} guided modes')
    print(f'{md[0].neff.real:g} symmetric mode effective index')
    print(f'{md[1].neff.real:g} antisymmetric mode effective index')
    print(f'{md.couplinglength():g}mm coupling length')
    if plot:
        Ls = wrange(0,30,0.1)
        Wave(md.crossover(Ls),Ls).plot(x='coupling length (mm)',y='crossover transmission',grid=1,xlim='f',save=f'ln rpe directional coupler crossover vs coupling length')
        md.plot(x='width (µm)',y='depth (µm)',save=f'mode profile, ln rpe directional coupler mode{md.modenum} at {λ}nm')

if __name__ == '__main__':
    plot = 0
    step = 0.5
    corningpmfiberexample(λ=1064,fiber=980,step=step,plot=plot)
    ktpwaveguideexample(pol='v',step=step,plot=plot)
    ktpwaveguideexample(pol='h',step=step,plot=plot)
    ridgewaveguideexample(step=step,plot=plot)
    rpewaveguideexample(step=step,plot=plot)
    rpedirectionalcoupler(step=step,plot=plot)
