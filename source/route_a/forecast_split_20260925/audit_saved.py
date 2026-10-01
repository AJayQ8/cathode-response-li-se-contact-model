"""Read-only independent arithmetic/fit audit; never integrates or refits."""
from pathlib import Path
import hashlib
import json
import numpy as np
from shared_compute import checkpoint_guard

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
REV=HERE.parent


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def direct(x,f):
    omega=2j*np.pi*f
    z=np.zeros(len(f),complex)
    for o in (0,3):
        r,fc,n=10**x[o],10**x[o+1],x[o+2]
        q=1/(r*(2*np.pi*fc)**n)
        z+=1/(1/r+q*omega**n)
    qt=1/(10**x[6]*(2*np.pi)**x[7])
    return z+1/(qt*omega**x[7])


def main():
    checkpoint_guard()
    m=read(HERE/'manifest.json'); summary=read(HERE/'summary.json')
    for p,h in m['sha256'].items():assert sha(ROOT/p)==h,p
    f=10**np.array(read(REV/'impedance8/spectra.json')['frequency_log10_hz'])
    objective_error=0.;reconstruction_error=0.;decomp_error=0.;quad_error=0.
    checked_fits=0; checked_starts=0; all_rows=[]; max_coordinate_effect_pp=0.
    for w in m['witnesses']:
        row=read(HERE/'cases'/(w['id']+'.json'));all_rows.append(row)
        assert row['x']==w['x'] and row['rstar']==w['rstar'] and row['setting']==w['setting']
        assert row['manifest_sha256']==sha(HERE/'manifest.json') and row['numerical_pass']
        source=read(REV/'weak33/cases'/('linear_'+w['id']+'.json'))
        primary={h['initial_cathode'][0]+h['history_cathode'][0]:h['rows'][0]['primary']['ratio'] for h in source['result']['histories']}
        independent={key:r['true_ratio_t0'] for key,r in row['cells'].items()}
        # Each 4-term interaction error is bounded by the sum of all cell differences.
        max_coordinate_effect_pp=max(max_coordinate_effect_pp,100*sum(abs(primary[k]-independent[k]) for k in primary))
        for field in ['fitted_ratio','true_ratio_t0']:
            a,b,c,d=[row['cells'][k][field] for k in ['PP','PN','NP','NN']]
            expected=dict(total=100*(a-d),history_at_N_initial=100*(c-d),initial_at_N_history=100*(b-d),
                          interaction=100*(a-b-c+d),history_at_P_initial=100*(a-b),initial_at_P_history=100*(a-c),
                          history_symmetric=50*(a-b+c-d),initial_symmetric=50*(a-c+b-d))
            got=row['decompositions'][field]['components_pp']
            decomp_error=max(decomp_error,max(abs(got[k]-v) for k,v in expected.items()))
            assert abs(expected['total']-expected['history_at_N_initial']-expected['initial_at_N_history']-expected['interaction'])<=1e-10
            assert abs(expected['total']-expected['history_symmetric']-expected['initial_symmetric'])<=1e-10
        targets=[r for r in row['cells'].values() if not r['reused']]
        targets += [c['readout'] for c in row['adapter_controls']]
        for r in targets:
            p=HERE/r['fit_path'];assert sha(p)==r['fit_sha256'];fit=read(p)
            observed=np.array(r['re_ohm_cm2'])-1j*np.array(r['minus_im_ohm_cm2'])
            z16=np.array(r['direct16_re_ohm_cm2'])-1j*np.array(r['direct16_minus_im_ohm_cm2'])
            qe=float(np.max(np.abs(observed-z16)/np.abs(z16)));quad_error=max(quad_error,qe);assert qe<=1e-8
            for s in fit['starts']:
                z=direct(np.array(s['x']),f);err=(z-observed)/np.abs(observed)
                rr=np.r_[err.real,err.imag]
                if fit['task']['loss']=='linear':cost=float(np.dot(rr,rr)/2)
                else:
                    u=(rr/.02)**2
                    cost=float(.02**2*np.sum(u/(np.sqrt(1+u)+1)))
                ce=abs(cost-s['objective']);objective_error=max(objective_error,ce)
                assert ce<=max(1e-13,1e-8*abs(s['objective']))
                checked_starts+=1
            best=min(fit['starts'],key=lambda s:s['objective'])
            assert best['start_index']==fit['best_start'] and best['success']
            x=np.array(best['x']);rh=10**x[0 if x[1]>=x[4] else 3]
            assert abs(rh/r['fitted_reference']-r['fitted_ratio'])<=1e-12
            saved=np.array(fit['predicted_re_ohm_cm2'])-1j*np.array(fit['predicted_minus_im_ohm_cm2'])
            re=float(np.max(np.abs(direct(x,f)-saved)/np.maximum(1.,np.abs(saved))))
            reconstruction_error=max(reconstruction_error,re);assert re<=1e-11
            checked_fits+=1
        checkpoint_guard()
    stats_error=0.
    for field in ['fitted_ratio','true_ratio_t0']:
        for key,item in summary['all_witnesses'][field].items():
            if key=='symmetric_history_fraction':continue
            vals=np.array([r['decompositions'][field]['components_pp'][key] for r in all_rows])
            for stat,val in [('minimum',min(vals)),('median',np.median(vals)),('maximum',max(vals))]:
                stats_error=max(stats_error,abs(item[stat]-val))
    assert len(all_rows)==50 and checked_fits==104 and checked_starts==624 and decomp_error<=1e-10 and stats_error<=1e-12
    receipt=dict(passed=True,witnesses=50,new_crossed_readouts=100,repeated_adapter_controls=4,
                 all_start_objectives_checked=checked_starts,maximum_objective_difference=objective_error,
                 maximum_angular_admittance_reconstruction_error=reconstruction_error,
                 maximum_quadrature_error=quad_error,maximum_decomposition_difference_pp=decomp_error,
                 maximum_summary_difference_pp=stats_error,
                 maximum_instantaneous_four_cell_coordinate_discrepancy_bound_pp=max_coordinate_effect_pp,
                 manifest_sha256=sha(HERE/'manifest.json'),auditor_sha256=sha(Path(__file__)),
                 discharge_integrations=0,kinetic_refits=0,scientific_sign_gate=False)
    (HERE/'audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
