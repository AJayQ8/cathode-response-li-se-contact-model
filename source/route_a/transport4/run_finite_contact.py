"""Finite-contact-resistance electrical component, separate from ideal contacts."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve
from shared_compute import checkpoint_guard
from run_transport import base_operator, contact_mask, solve as ideal_solve

HERE = Path(__file__).resolve().parent


def save(name, value):
    path = HERE/name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def solve(n, active, conductance, *, conductivity=1., current=1., metal=None):
    a = np.asarray(active, dtype=float)
    assert a.shape == (n,) and np.all((a >= 0) & (a <= 1))
    assert conductance >= 0 and conductivity > 0 and current != 0
    if conductance == 0 or not np.any(a):
        return {'status':'disconnected'}, None
    dx = dy = 1/n
    local = conductance*a
    b = local/(1+local*dy/(2*conductivity))
    vb = np.zeros(n) if metal is None else np.asarray(metal)
    diagonal = np.zeros(n*n)
    diagonal[:n] = dx*b
    operator = conductivity*base_operator(n,n,1.) + sparse.diags(diagonal,format='csc')
    rhs = np.zeros(n*n)
    rhs[-n:] = current*dx
    rhs[:n] += dx*b*vb
    potential = spsolve(operator,rhs).reshape(n,n)
    q = b*(potential[0]-vb)
    surface = potential[0]-q*dy/(2*conductivity)
    top = potential[-1]+current*dy/(2*conductivity)
    resistance = float(np.mean(top)/current)
    bulk = conductivity*np.sum((potential-np.roll(potential,-1,axis=1))**2)
    bulk += conductivity*np.sum(np.diff(potential,axis=0)**2)
    boundary_half_cells = (np.sum(q*q)*dx + current*current)*dy/(2*conductivity)
    interface = float(np.sum(np.divide(q*q,local,out=np.zeros(n),where=local>0))*dx)
    dissipation = float(bulk+boundary_half_cells+interface)
    work = float(current*dx*np.sum(top)-dx*np.sum(q*vb))
    residual = operator @ potential.ravel()-rhs
    active_q = np.divide(q,a,out=np.zeros(n),where=a>0)
    result = {'status':'solved','n':n,'beta':conductance/conductivity,
              'conductance':conductance,'conductivity':conductivity,'applied_current':current,
              'mean_active_fraction':float(np.mean(a)),
              'active_fraction_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
              'resistance_over_bulk':resistance*conductivity,
              'resistance_over_full_contact':resistance/(1/conductivity+1/conductance),
              'maximum_nominal_current_over_applied':float(np.max(q/current)),
              'maximum_active_current_over_applied':float(np.max(active_q/current)),
              'interface_potential_range': [float(np.min(surface)),float(np.max(surface))],
              'current_relative_error':abs(float(np.sum(q)*dx)-current)/abs(current),
              'power_relative_error':abs(dissipation-work)/max(abs(work),1e-30),
              'linear_residual_relative_l2':float(np.linalg.norm(residual)/np.linalg.norm(rhs)),
              'zero_gap_current':bool(np.all(q[a==0]==0)),
              'robin_law_maximum_error':float(np.max(abs(q-local*(surface-vb)))),
              'bulk_dissipation':float(bulk),'boundary_half_cell_dissipation':float(boundary_half_cells),
              'interface_dissipation':interface,'total_dissipation':dissipation,'boundary_work':work}
    return result,(potential,q)


def conservative(r):
    return all(r[k] <= 1e-9 for k in
               ['current_relative_error','power_relative_error','linear_residual_relative_l2']) and r['zero_gap_current']


def geometries(n):
    rows=[]
    for fraction,patches in [(.5,1),(.5,4),(.25,4)]:
        pitch=n//patches
        count=int(pitch*fraction)
        assert n%patches==0 and count==pitch*fraction
        one=np.zeros(pitch)
        start=(pitch-count)//2
        one[start:start+count]=1.
        a=np.tile(one,patches)
        # n=32 limiting-profile fixtures need no edge integral. Two cells
        # per small patch cannot represent the fixed quarter-width windows.
        edge=contact_mask(n,fraction,patches)[1] if count%4==0 else None
        rows.append((f'area{fraction:g}_patches{patches}',a,edge))
    x=(np.arange(n)+.5)/n
    rows.append(('smooth_active_fraction',.5+.4*np.cos(2*np.pi*x),None))
    return rows


def consistency():
    exact=[]
    for fraction in [.25,.6,1.]:
        for k,j,g in [(.3,.2,.6),(1.,1.,.1),(3.,-4.,30.)]:
            r,fields=solve(32,np.full(32,fraction),g,conductivity=k,current=j)
            y=(np.arange(32)+.5)/32
            truth=j/(g*fraction)+j*y/k
            error=float(np.max(abs(fields[0]-truth[:,None])))
            expected=1/k+1/(g*fraction)
            r['maximum_field_error']=error
            r['pass']=error <= 1e-9*(1+np.max(abs(truth))) and abs(r['resistance_over_bulk']-k*expected) <= 1e-9*(1+k*expected) and conservative(r)
            r['pass']=bool(r['pass'])
            exact.append(r)
    mms=[]
    for n in [16,32,64,128]:
        checkpoint_guard()
        x,y=(np.arange(n)+.5)/n,(np.arange(n)+.5)/n
        k=2*np.pi
        metal=.03*(1+k*np.tanh(k)/3)*np.cos(k*x)
        r,fields=solve(n,np.ones(n),3.,metal=metal)
        truth=1/3+y[:,None]+.03*np.cos(k*x)[None,:]*np.cosh(k*(1-y[:,None]))/np.cosh(k)
        r['rms_error']=float(np.sqrt(np.mean((fields[0]-truth)**2)))
        if mms:
            r['order']=float(np.log2(mms[-1]['rms_error']/r['rms_error']))
        mms.append(r)
    limits=[]
    for name,a,edge in geometries(32):
        r,fields=solve(32,a,.001)
        equal=a/np.mean(a)
        error=float(np.max(abs(fields[1]-equal))/np.max(equal))
        limits.append({'name':name,'limit':'equal_conductance','beta':.001,
                       'maximum_relative_profile_error':error,'pass':error<=.01 and conservative(r),
                       'conservation':{k:r[k] for k in ['current_relative_error','power_relative_error','linear_residual_relative_l2']}})
    for name,a,edge in geometries(64):
        if edge is None:
            continue
        r,fields=solve(64,a,1e6)
        original,orig_fields=ideal_solve(64,64,1.,a.astype(bool))
        re=abs(r['resistance_over_bulk']-original['effective_resistance'])/original['effective_resistance']
        qe=float(np.linalg.norm(fields[1]-orig_fields[1])/np.linalg.norm(orig_fields[1]))
        limits.append({'name':name,'limit':'ideal_contact','beta':1e6,
                       'resistance_relative_error':re,'current_profile_relative_l2':qe,
                       'pass':re<=1e-4 and qe<=.001 and conservative(r)})
    disconnected=[solve(32,np.zeros(32),1.)[0]['status']=='disconnected',
                  solve(32,np.ones(32),0.)[0]['status']=='disconnected']
    passed=all(r['pass'] for r in exact+limits) and all(disconnected)
    passed=passed and mms[-1]['rms_error']<=1e-4 and all(r['order']>=1.8 for r in mms[-2:]) and all(conservative(r) for r in mms)
    output={'uniform_cases':exact,'manufactured_solution':mms,'limits':limits,
            'disconnection_cases':disconnected,'all_pass':bool(passed)}
    save('finite_consistency.json',output)
    return output


def summary(cases):
    comparisons=[]
    for beta in [.1,1.,10.,100.]:
        for name,_,edge in geometries(64):
            group=sorted([r for r in cases if r['beta']==beta and r['name']==name],key=lambda r:r['n'])
            for a,b in zip(group,group[1:]):
                re=abs(a['resistance_over_bulk']-b['resistance_over_bulk'])/b['resistance_over_bulk']
                pe=abs(a['maximum_nominal_current_over_applied']-b['maximum_nominal_current_over_applied'])/b['maximum_nominal_current_over_applied']
                ee=None if edge is None else abs(a['edge_current_share']-b['edge_current_share'])
                comparisons.append({'name':name,'beta':beta,'coarse_n':a['n'],'fine_n':b['n'],
                                    'resistance_relative_change':re,'maximum_current_relative_change':pe,
                                    'edge_share_absolute_change':ee,'resistance_pass':re<=.01,
                                    'maximum_current_pass':pe<=.01,'edge_share_pass':None if ee is None else ee<=.01})
    finest=[r for r in comparisons if r['fine_n']==256]
    out={'cases':len(cases),'all_conservation_pass':all(conservative(r) for r in cases),
         'comparisons':comparisons,'finest_resistance_pass_count':sum(r['resistance_pass'] for r in finest),
         'finest_peak_current_pass_count':sum(r['maximum_current_pass'] for r in finest),
         'finest_edge_share_pass_count':sum(bool(r['edge_share_pass']) for r in finest),
         'finest_edge_share_comparison_count':sum(r['edge_share_pass'] is not None for r in finest),
         'calibrated_conductance_or_geometry':False,'cathode_amplification_predicted':False}
    save('finite_summary.json',out)
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    names=['run_finite_contact.py','FINITE_CONTACT_PLAN.md','run_transport.py']
    manifest={'sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names},
              'numpy':np.__version__,'scipy':scipy.__version__}
    checkpoint_guard()
    if args.resume:
        assert json.loads((HERE/'finite_manifest.json').read_text())==manifest
        checks=json.loads((HERE/'finite_consistency.json').read_text())
        cases=json.loads((HERE/'finite_cases.json').read_text())
    else:
        assert not (HERE/'finite_consistency.json').exists(), 'Preserve completed/failed checks.'
        save('finite_manifest.json',manifest)
        checks=consistency()
        cases=[]
    if not checks['all_pass']:
        raise RuntimeError('Consistency gate failed; preserve the result and reassess.')
    for n in ([64] if args.pilot else [64,128,256]):
        for beta in [.1,1.,10.,100.]:
            for name,a,edge in geometries(n):
                if any(r['name']==name and r['beta']==beta and r['n']==n for r in cases):
                    continue
                checkpoint_guard()
                tick=time.perf_counter()
                r,fields=solve(n,a,beta)
                r.update({'name':name,'elapsed_seconds':time.perf_counter()-tick,
                          'edge_current_share':None if edge is None else float(np.sum(fields[1][edge])/np.sum(fields[1]))})
                if n==256:
                    r['surface_profile']={'x':((np.arange(n)+.5)/n).tolist(),
                                          'active_fraction':a.tolist(),'current':fields[1].tolist()}
                cases.append(r)
                save('finite_cases.json',cases)
                print(json.dumps({k:r[k] for k in ['name','beta','n','resistance_over_bulk','maximum_nominal_current_over_applied','elapsed_seconds']}),flush=True)
                del fields
    s=summary(cases)
    print(json.dumps({k:v for k,v in s.items() if k!='comparisons'}))


if __name__=='__main__':
    main()
