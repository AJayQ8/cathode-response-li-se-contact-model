"""Version-2 fixed-m projections with cross-attempt exact-vector deduplication."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, math, os, time
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from shared_compute import checkpoint_guard

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent
ROOT=HERE.parents[4]
J,Q=1.2,.30
BUDGET=.001

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
kernel=module("m_project_kernel",HERE/"m_kernel.py")
source=kernel.source
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
  if isinstance(v,complex):return [float(v.real) if math.isfinite(v.real) else None,
                                  float(v.imag) if math.isfinite(v.imag) else None]
  if isinstance(v,np.ndarray):return plain(v.tolist())
  if isinstance(v,np.generic):return plain(v.item())
  if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [plain(x) for x in v]
  if isinstance(v,float) and not math.isfinite(v):return None
  return v
def save(p,v):
  checkpoint_guard();p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".tmp-"+str(os.getpid()))
  with tmp.open("w") as f:f.write(json.dumps(plain(v),indent=2,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
  os.replace(tmp,p)

@lru_cache(maxsize=None)
def raw_wave(cathode):
  mech=PAPER/"mechanics3";rows=list(csv.DictReader((mech/"waveforms.csv").open()))
  meta=next(x for x in read(mech/"waveform_summary.json") if x["cathode"]==cathode)
  data=[r for r in rows if r["cathode"]==cathode and r["quantity"]=="pressure_kpa"]
  tt=np.asarray([float(r["time_h"]) for r in data]);pp=np.asarray([float(r["value"]) for r in data])
  t0=float(meta["charge_end_voltage_max"]["time_h"]);end=float(meta["common_end_h"]);p0=float(np.interp(t0,tt,pp))
  return tt,pp,t0,p0,end
def raw_delta(cathode,q):
  tt,pp,t0,p0,end=raw_wave(cathode);assert t0+q/.12<=end+1e-10
  return float((np.interp(t0+q/.12,tt,pp)-p0)/1000.)

def fit_id(m,e,s,nul):return f"m{str(m).replace('.','p')}_{e}_{s['witness_id']}_{'eta0' if nul else 'free'}"
def candidate_key(candidate):
  x=np.asarray(candidate["x"],dtype="<f8").copy()
  x[x==0.]=0. # +0 and -0 are the same exact numerical coordinate for deduplication.
  return hashlib.sha256(x.tobytes()).hexdigest()
def source_reference(sweep,e,c):
  rec=next(r for r in read(PAPER/"sweep35/controls.json")["records"] if r["extraction"]==e and r["cathode"]==c)
  assert rec["passed"]
  return rec

def project_candidate(attempt,candidate,fit_record):
  m=fit_record["fixed_m"];e=fit_record["extraction"];seed=fit_record["seed"];x=np.asarray(candidate["x"],float)
  sweep=module("m_fixedm_sweep35",PAPER/"sweep35/run_sweep.py")
  rstar,protocols=source.data(e,"balance_aligned")
  f=sweep.frequency();schedule,duration_s=sweep.schedule(f,"high_to_low")
  assert len(f)==68 and abs(duration_s-106.9561255903888)<=1e-10
  n8,w8=np.polynomial.legendre.leggauss(8);n16,w16=np.polynomial.legendre.leggauss(16)
  cells=[]
  L,H0=10.**x[:2];eta,b,A=x[2:]
  for cathode in ["P-LCO","N-LCO"]:
    checkpoint_guard()
    p=next(p for p in protocols if p["cathode"]==cathode and p["rate_c"]==2.)
    trial=dict(p,J=J,t1=Q/J,t2=0.,rest=0.)
    loading=source.pressure.domain(trial,float(eta))
    state0,z0,a0=kernel.initial_state(x,rstar,p["R_ch"])
    cell=dict(cathode=cathode,R_ch=p["R_ch"],q0=p["q0"],rstar=rstar,loading=loading,
              initial_z=z0,initial_a=a0,projection=None,rest_trajectory=None,spectrum=None,fit=None)
    if state0 is None or not loading["valid"]:
      cell.update(available=False,reason=loading.get("reason") or "initial contact invalid");cells.append(cell);continue
    state,stopped,elapsed,proj=kernel.advance_m(state0,Q/J,x,cathode,p["q0"],J,m)
    ae=A*math.sqrt(max(0.,float(state[0])));charge=J*elapsed
    ratio=None if stopped else float((b*rstar+A*(1-b)*rstar/ae)/p["R_ch"])
    model_proj=dict(stopped=stopped,elapsed_h=elapsed,charge_mAh_cm2=charge,contact=ae,
      ratio=ratio,w=float(state[0]),projection=proj,height_um=L*ae,
      stock_mAh_cm2=float(state[0]/(2.*kernel.U/(L*A*A))+charge))
    # Independently integrate recession and replenished volume from raw pressure/time samples.
    V0=L*a0*a0/2.;y0=np.array([L*(1-a0),0.])
    # Explicitly compare model interpolation with the untouched source-time
    # pressure samples across this discharge segment.
    tt_src,pp_src,tzero,pzero,tend_src=raw_wave(cathode)
    pressure_errors=[]
    for qv in [p["q0"],p["q0"]+Q]:
      pressure_errors.append(abs((2.+eta*raw_delta(cathode,qv))-
                                 (2.+eta*float(source.pressure_increment(cathode,qv)))))
    # Both the model and raw waveform use charge coordinates from this source's
    # charge-end origin; compare knots directly over the shifted interval.
    raw_qknots=(tt_src[(tt_src>=tzero)&(tt_src<=tend_src)]-tzero)*.12
    for qv in raw_qknots:
      qv=float(qv)
      if p["q0"]<=qv<=p["q0"]+Q:
        pressure_errors.append(abs((2.+eta*raw_delta(cathode,qv))-
                                   (2.+eta*float(source.pressure_increment(cathode,qv)))))
    pressure_error=max(pressure_errors,default=0.)
    def rhs(t,y):
      a=max(min(1-y[0]/L,1.),A*kernel.ZMIN/4.)
      pressure=2.+eta*raw_delta(cathode,p["q0"]+J*t)
      recovery=H0*L*A**(m+1.)*(pressure/2.)**m
      return [kernel.U*J/a-recovery*(1-a)/a**m,recovery*(1-a)*a**(1-m)]
    def event(t,y):return y[0]-L*(1-A*kernel.ZMIN)
    event.terminal,event.direction=True,1
    sol=solve_ivp(rhs,(0.,Q/J),y0,method="Radau",rtol=1e-11,atol=1e-13,
      first_step=min(Q/J,1e-4),events=event)
    if not sol.success:
      cell.update(available=False,reason="physical projection Radau failure: "+sol.message);cells.append(cell);continue
    yp=sol.y[:,-1];pstop=bool(len(sol.t_events[0]));pel=float(sol.t[-1]);pq=J*pel
    ap=max(0.,min(1.,1-yp[0]/L));vp=L*ap*ap/2.
    pratio=None if pstop else float((b*rstar+A*(1-b)*rstar/ap)/p["R_ch"])
    ledger=abs(vp-V0+kernel.U*pq-yp[1]);ratio_delta=None if ratio is None or pratio is None else abs(ratio-pratio)
    repl_delta=abs(yp[1]-(float(state[0])-float(state0[0])+2.*kernel.U/(L*A*A)*charge)*L*A*A/2.)
    proj_audit=dict(stopped=pstop,elapsed_h=pel,stripped_charge_mAh_cm2=pq,contact=ap,ratio=pratio,
      recession_um=float(yp[0]),replenished_um=float(yp[1]),volume_um=vp,initial_volume_um=V0,
      ledger_um=ledger,ratio_difference=ratio_delta,contact_difference=abs(ae-ap),
      replenishment_difference_um=repl_delta,projection=proj,
      pressure_interpolation_error_mpa=pressure_error,
      passed=bool(stopped==pstop and not pstop and pressure_error<=1e-10 and
        abs(elapsed-pel)<=1e-6 and abs(charge-pq)<=1e-6 and
        abs(ae-ap)<=1e-7 and (ratio_delta is None or ratio_delta<=1e-7) and
        ledger<=1e-6*max(1.,vp,abs(yp[1])) and repl_delta<=1e-5*max(1.,abs(yp[1])) and
        proj<=1e-7*max(1.,1/(A*A)) and 0.<ap<=1. and L*ap<=40.+1e-10 and
        vp/kernel.U+pq<=40./kernel.U+1e-7 and yp[1]>=-1e-7))
    cell["projection"]=dict(model=model_proj,independent=proj_audit)
    if not proj_audit["passed"]:
      cell.update(available=False,reason="independent projection physical gate failed");cells.append(cell);continue

    qstop=p["q0"]+Q
    p_raw=2.+eta*raw_delta(cathode,qstop);p_fast=2.+eta*source.pressure_increment(cathode,qstop)
    pdiff=abs(p_raw-p_fast);gamma=H0*A**(m+1.)*(p_raw/2.)**m;T=duration_s/3600.
    def arhs(t,y):
      aa=max(float(y[0]),A*kernel.ZMIN/4.);return [gamma*(1-aa)/aa**m]
    arest=solve_ivp(arhs,(0.,T),[ae],method="Radau",rtol=1e-11,atol=1e-13,
      first_step=min(T,1e-4),dense_output=True)
    query=sorted(set([0.,duration_s]+[r["measure_start_s"] for r in schedule]+[r["measure_end_s"] for r in schedule]))
    def grhs(t,y):
      aa=max(min(1-y[0]/L,1.),A*kernel.ZMIN/4.);rate=H0*L*A**(m+1.)*(p_raw/2.)**m
      return [-rate*(1-aa)/aa**m,rate*(1-aa)*aa**(1-m)]
    # Continue the independently integrated discharge state, including its
    # replenished-volume ledger, through the fixed-pressure readout interval.
    g0=np.asarray(yp,dtype=float).copy()
    grest=solve_ivp(grhs,(0.,T),g0,method="Radau",rtol=1e-11,atol=1e-13,
      first_step=min(T,1e-4),dense_output=True)
    nodes=[]
    if arest.success and grest.success:
      for ts in query:
        aa=float(arest.sol(ts/3600.)[0]);gg=grest.sol(ts/3600.);ag=max(0.,min(1.,1-float(gg[0])/L))
        rr=float((b*rstar+A*(1-b)*rstar/aa)/p["R_ch"]);rg=float((b*rstar+A*(1-b)*rstar/ag)/p["R_ch"])
        vv=L*ag*ag/2.;led=abs(vv-V0+kernel.U*pq-float(gg[1]))
        ok=bool(abs(aa-ag)<=1e-7 and abs(rr-rg)<=1e-7 and led<=1e-6*max(1.,vv,abs(gg[1]))
          and 0.<ag<=1. and L*ag<=40.+1e-10 and gg[1]>=-1e-7)
        nodes.append(dict(time_s=ts,model_contact=aa,physical_contact=ag,contact_difference=abs(aa-ag),
          model_ratio=rr,physical_ratio=rg,ratio_difference=abs(rr-rg),remaining_volume_um=vv,
          replenished_um=float(gg[1]),ledger_um=led,stripped_charge_mAh_cm2=pq,passed=ok))
    rest_ok=bool(arest.success and grest.success and len(nodes)==len(query) and all(n["passed"] for n in nodes) and pdiff<=1e-10)
    rest_record=dict(pressure_model_mpa=p_fast,pressure_raw_mpa=p_raw,pressure_difference_mpa=pdiff,
      gamma_per_hour=gamma,nodes=nodes,passed=rest_ok)
    cell["rest_trajectory"]=rest_record
    if not rest_ok:
      cell.update(available=False,reason="independent fixed-m rest trajectory check failed");cells.append(cell);continue

    template=sweep.make_template(e,cathode,p["R_ch"]);ref=source_reference(sweep,e,cathode)
    assert template==ref["template"]
    def make_spectrum(nodes,weights,direct):
      result=[]
      for k,row in enumerate(schedule):
        u,v=row["measure_start_s"],row["measure_end_s"];times=(u+v)/2+nodes*(v-u)/2
        contacts=np.asarray([float(arest.sol(t/3600.)[0]) for t in times])
        resistances=b*rstar+A*(1-b)*rstar/contacts
        vals=[sweep.forward(np.asarray([f[k]]),np.asarray([rh]),template,"fixed_q",direct=direct)[0] for rh in resistances]
        result.append(np.dot(weights,np.asarray(vals))/2.)
      return np.asarray(result,complex)
    z8=make_spectrum(n8,w8,False);z16=make_spectrum(n16,w16,True)
    qerr=float(np.max(np.abs(z8-z16)/np.abs(z16)))
    sid=f"{attempt}_{candidate_key(candidate)}_{cathode[0]}"
    fittask=dict(spectrum_id=sid,topology="source",loss=sweep.loss_for(e),fraction=0.)
    spectrum=dict(re_ohm_cm2=z8.real.tolist(),minus_im_ohm_cm2=(-z8.imag).tolist(),
      source_bulk_li_r=template["Rref"],source_cathode_r=template["slow_r"])
    rawdir=HERE/"circuit_fits_v3"/sid;rawpath=rawdir/(sweep.circuit.task_id(fittask)+".json")
    reused_raw_fit=rawpath.exists()
    if reused_raw_fit:
      fit=read(rawpath)
      assert fit["task"]==fittask and fit["id"]==sweep.circuit.task_id(fittask)
    else:
      prev=sweep.circuit.OUT
      try:
        sweep.circuit.OUT=rawdir;sweep.circuit.fit_one(fittask,spectrum,f)
      finally:sweep.circuit.OUT=prev
      fit=read(rawpath)
    best=fit["best"];Rfit=best["parameters"]["fast"]["r_ohm_cm2"]
    cell["spectrum"]=dict(template=template,stationary_reference_path=ref["fit_path"],
      stationary_reference=ref["fitted_reference"],loss=sweep.loss_for(e),frequency_hz=f.tolist(),
      schedule=schedule,spectrum8=z8.tolist(),spectrum16=z16.tolist(),quadrature_relative_error=qerr,
      quadrature_pass=qerr<=1e-8,duration_s=duration_s)
    fit_ok=bool(best["success"] and len(fit["starts"])==6 and qerr<=1e-8)
    cell["fit"]=dict(output=fit,all_six_starts_present=len(fit["starts"])==6,reused_raw_fit=reused_raw_fit,
      failed_starts=sum(not r["success"] for r in fit["starts"]),selected_success=bool(best["success"]),
      fast_R_ohm_cm2=Rfit,normalized_Rh=Rfit/ref["fitted_reference"],
      current_stop_normalized_Rh=ratio,readout_end_normalized_Rh=nodes[-1]["physical_ratio"],
      stationary_reference=ref["fitted_reference"],raw_path=str(rawpath.relative_to(HERE)))
    cell["available"]=fit_ok;cell["reason"]=None if fit_ok else "quadrature/start/selected-fit gate failed"
    cells.append(cell)
  out=dict(attempt=attempt,labels=candidate["labels"],fixed_m=m,extraction=e,seed_id=seed["witness_id"],
    null=bool(x[2]==0.),x=x.tolist(),cells=cells,manifest_sha256=sha(HERE/"manifest.json"),
    within_window_pressure_history_effect="zero by construction" if x[2]==0. else "nonzero transfer fit parameter",
    starting_state_difference_retained=True)
  if len(cells)==2 and all(c.get("fit") for c in cells):
    P,N=cells
    contrast=P["fit"]["normalized_Rh"]-N["fit"]["normalized_Rh"]
    true0=P["fit"]["current_stop_normalized_Rh"]-N["fit"]["current_stop_normalized_Rh"]
    out.update(fitted_P_minus_N_contrast=contrast,instantaneous_current_stop_contrast=true0,
      signed_acquisition_bias=contrast-true0,bias_budget=BUDGET,
      bias_budget_pass=bool(abs(contrast-true0)<=BUDGET),
      all_selected_fits_success=all(c["fit"]["selected_success"] for c in cells),
      physical_trajectory_pass=all(c["rest_trajectory"]["passed"] and c["projection"]["independent"]["passed"] for c in cells),
      all_quadrature_pass=all(c["spectrum"]["quadrature_pass"] for c in cells),
      numerical_pass=all(c["available"] and c["rest_trajectory"]["passed"] and
        c["projection"]["independent"]["passed"] for c in cells))
  else:
    out.update(fitted_P_minus_N_contrast=None,instantaneous_current_stop_contrast=None,
      signed_acquisition_bias=None,bias_budget=BUDGET,bias_budget_pass=False,numerical_pass=False)
  return out


def expected_fit_ids(fit_manifest):
  return {fit_id(m,seed["extraction"],seed,null)
          for m in fit_manifest["exponents"] for seed in fit_manifest["seeds"] for null in (False,True)}

def accuracy_requalification_inputs():
  """Validate the separate no-refit accuracy receipts before admitting four exact vectors."""
  folder=HERE/"accuracy_requalification_v1"
  audit_path=folder/"AUDIT.json";manifest_path=folder/"requalification_manifest.json"
  targets_path=folder/"targets.json";execution_path=folder/"accuracy_all_execution.json"
  audit=read(audit_path);manifest=read(manifest_path);target_index=read(targets_path);execution=read(execution_path)
  assert audit["passed"] and audit["complete_four_target_count"]==4 and not audit["failures"]
  assert audit["target_index_sha256"]==sha(targets_path)
  assert audit["requalification_manifest_sha256"]==sha(manifest_path)
  assert audit["result_hashes"] and len(audit["result_hashes"])==4
  assert execution["phase"]=="all" and execution["target_count"]==4
  rows=[]
  for entry in target_index["targets"]:
    tid=entry["attempt_id"]+f"__c{entry['candidate_index']}"
    result_path=folder/"results"/(tid+".json");result=read(result_path)
    rel=str(result_path.relative_to(folder))
    assert sha(result_path)==audit["result_hashes"].get(rel)
    assert result["newly_qualified"] and result["x"]==entry["x"]
    assert result["vector_sha256"]==entry["vector_sha256"]
    fit_path=HERE/entry["fit_path"];fit=read(fit_path)
    assert sha(fit_path)==entry["fit_sha256"]
    candidate=fit["candidates"][entry["candidate_index"]]
    assert candidate["x"]==entry["x"] and not candidate["qualified_compatible"]
    provenance=dict(attempt_id=entry["attempt_id"],fit_path=entry["fit_path"],fit_sha256=entry["fit_sha256"],
      candidate_index=entry["candidate_index"],labels=entry["labels"],seed_id=entry["seed_id"],
      null=bool(entry["null"]),qualified_compatible=True,
      max_abs_error=entry["original_max_abs_error"],all14_SSE=entry["original_all14_SSE"],
      physical_fast_pass=True,domain_pass=True,screen_margin=entry["original_screen_margin"],
      numeric_disagreement=entry["original_ratio_disagreement"],
      qualification_basis="accuracy_requalification_v1",original_fit_qualified_compatible=False,
      accuracy_target_id=tid,accuracy_target_sha256=sha(targets_path),
      accuracy_manifest_sha256=sha(manifest_path),accuracy_audit_sha256=sha(audit_path),
      accuracy_result_sha256=sha(result_path))
    rows.append(dict(target=entry,provenance=provenance,result_path=rel,result_sha256=sha(result_path)))
  return dict(folder=folder,audit_path=audit_path,manifest_path=manifest_path,targets_path=targets_path,
    execution_path=execution_path,audit=audit,manifest=manifest,target_index=target_index,rows=rows)

def freeze_projection_inputs():
  fit_manifest_path=HERE/"manifest.json";fit_manifest=read(fit_manifest_path);fit_manifest_sha=sha(fit_manifest_path)
  controls=read(HERE/"controls.json")
  assert controls["passed"] and controls["manifest_sha256"]==fit_manifest_sha
  fits=sorted((HERE/"fits").glob("*.json"));expected=expected_fit_ids(fit_manifest)
  observed={p.stem for p in fits}
  assert observed==expected and len(fits)==24,(len(fits),sorted(expected-observed),sorted(observed-expected))
  fit_records={p.stem:read(p) for p in fits}
  fit_hashes={str(p.relative_to(HERE)):sha(p) for p in fits}
  for fid,fit in fit_records.items():
    assert fit["id"]==fid and fit["manifest_sha256"]==fit_manifest_sha
    assert fit["fixed_m"] in (5.9,7.3) and fit["extraction"] in ("ordinary","robust","published")
    assert fit["optimizer_caps"]==dict(maxiter=60,max_uncached_forward_evaluations=200)
    assert fit["optimizer_forward_evaluations"]<=200
  groups={};excluded=[]
  for fid in sorted(fit_records):
    fit=fit_records[fid];fhash=fit_hashes[str(Path("fits")/(fid+".json"))]
    for ci,cand in enumerate(fit["candidates"]):
      provenance=dict(attempt_id=fid,fit_path=str(Path("fits")/(fid+".json")),fit_sha256=fhash,
        candidate_index=ci,labels=cand["labels"],seed_id=fit["seed"]["witness_id"],null=bool(fit["null"]),
        qualified_compatible=bool(cand["qualified_compatible"]),max_abs_error=cand.get("max_abs_error"),
        all14_SSE=cand.get("all14_SSE"),physical_fast_pass=cand.get("physical_fast_pass"),
        domain_pass=cand.get("domain_pass"),screen_margin=cand.get("screen_margin"),
        numeric_disagreement=cand.get("numeric_disagreement"))
      if not cand["qualified_compatible"]:
        excluded.append(dict(provenance=provenance,reason="candidate failed frozen physical/all14/screen qualification"));continue
      x=np.asarray(cand["x"],dtype="<f8").copy();assert x.shape==(5,)
      x[x==0.]=0.
      vector_sha=hashlib.sha256(x.tobytes()).hexdigest()
      group_key=(float(fit["fixed_m"]),fit["extraction"],vector_sha)
      if group_key not in groups:
        groups[group_key]=dict(target_id=f"m{str(fit['fixed_m']).replace('.','p')}_{fit['extraction']}_{vector_sha}",
          fixed_m=fit["fixed_m"],extraction=fit["extraction"],vector_sha256=vector_sha,
          x=cand["x"],canonical_attempt=fid,canonical_candidate_index=ci,canonical_basis="original_fit",
          provenance=[provenance])
      else:
        rec=groups[group_key]
        assert np.array_equal(np.asarray(rec["x"],float),np.asarray(cand["x"],float))
        rec["provenance"].append(provenance)
  accuracy=accuracy_requalification_inputs()
  for row in accuracy["rows"]:
    entry=row["target"];provenance=row["provenance"]
    x=np.asarray(entry["x"],dtype="<f8").copy();assert x.shape==(5,)
    x[x==0.]=0.;vector_sha=hashlib.sha256(x.tobytes()).hexdigest()
    assert vector_sha==entry["vector_sha256"]
    group_key=(float(entry["fixed_m"]),entry["extraction"],vector_sha)
    if group_key not in groups:
      groups[group_key]=dict(target_id=f"m{str(entry['fixed_m']).replace('.','p')}_{entry['extraction']}_{vector_sha}",
        fixed_m=entry["fixed_m"],extraction=entry["extraction"],vector_sha256=vector_sha,
        x=entry["x"],canonical_attempt=entry["attempt_id"],canonical_candidate_index=entry["candidate_index"],
        canonical_basis="accuracy_requalification_v1",provenance=[provenance])
    else:
      rec=groups[group_key]
      assert np.array_equal(np.asarray(rec["x"],float),x)
      rec["provenance"].append(provenance)
  targets=sorted(groups.values(),key=lambda r:r["target_id"])
  for t in targets:t["output_path"]=str(Path("projections_v3")/(t["target_id"]+".json"))
  index=dict(schema=2,fit_manifest_sha256=fit_manifest_sha,fit_records=fit_hashes,
    target_count=len(targets),qualified_provenance_count=sum(len(t["provenance"]) for t in targets),
    original_qualified_provenance_count=10,accuracy_requalified_provenance_count=4,
    exact_vector_target_count=len(targets),targets=targets,excluded_candidates=excluded,
    deduplication="within (fixed_m, extraction), exact float64 coordinate equality; signed zero canonicalized",
    candidates_are_forecast_eligible_only=True)
  index_path=HERE/"projection_index_v3.json";manifest_path=HERE/"projection_manifest_v3.json"
  if manifest_path.exists():
    assert index_path.exists(),"projection manifest exists without its index; preserve and inspect"
    assert read(index_path)==index,"refusing to alter frozen projection index"
    current=read(manifest_path)
    assert current["fit_manifest_sha256"]==fit_manifest_sha and current["index_sha256"]==sha(index_path)
    verify_projection_freeze()
    return current
  if index_path.exists():assert read(index_path)==index,"refusing to alter a preserved partial projection index"
  source_hashes={rel:digest for rel,digest in fit_manifest["sha256"].items()}
  for rel,digest in source_hashes.items():assert sha(ROOT/rel)==digest,rel
  qualification=read(HERE/"projection_qualification_v3.json")
  assert qualification["passed"] and qualification["projection_script_sha256"]==sha(HERE/"run_m_projection_v3.py")
  assert qualification["audit_script_sha256"]==sha(HERE/"audit_saved_m_v3.py")
  assert qualification["preflight_script_sha256"]==sha(HERE/"run_projection_preflight_v3.py")
  source_paths={
    "run_m_projection_v3.py":HERE/"run_m_projection_v3.py",
    "audit_saved_m_v3.py":HERE/"audit_saved_m_v3.py",
    "run_projection_preflight_v3.py":HERE/"run_projection_preflight_v3.py",
    "qualify_projection_v3.py":HERE/"qualify_projection_v3.py",
    "projection_qualification_v3.json":HERE/"projection_qualification_v3.json",
    "fit_manifest.json":fit_manifest_path,
    "controls.json":HERE/"controls.json",
    "sweep35_controls.json":PAPER/"sweep35/controls.json",
    "sweep35_manifest.json":PAPER/"sweep35/manifest.json",
    "m_kernel.py":HERE/"m_kernel.py",
    "sweep35_run_sweep.py":PAPER/"sweep35/run_sweep.py",
    "circuit_fitter.py":PAPER/"impedance8/run_circuit_probe.py",
    "mechanics_waveforms.csv":PAPER/"mechanics3/waveforms.csv",
    "mechanics_waveform_summary.json":PAPER/"mechanics3/waveform_summary.json",
    "accuracy_requalification_manifest.json":accuracy["manifest_path"],
    "accuracy_requalification_targets.json":accuracy["targets_path"],
    "accuracy_requalification_audit.json":accuracy["audit_path"],
    "accuracy_requalification_execution.json":accuracy["execution_path"]}
  for name,p in source_paths.items():
    if name=="fit_manifest.json":assert sha(p)==fit_manifest_sha
  reference_fits=[];sw=read(PAPER/"sweep35/controls.json")
  for rec in sw["records"]:
    p=PAPER/"sweep35"/rec["fit_path"]
    reference_fits.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),
      extraction=rec["extraction"],cathode=rec["cathode"],fitted_reference=rec["fitted_reference"],
      template=rec["template"]))
  if not index_path.exists():save(index_path,index)
  pm=dict(schema=2,version="projection-v3-cross-attempt-dedup",fit_manifest_sha256=fit_manifest_sha,
    fit_record_hashes=fit_hashes, index_path=str(index_path.relative_to(HERE)),index_sha256=sha(index_path),
    source_hashes=source_hashes,projection_source_hashes={k:sha(v) for k,v in source_paths.items()},
    stationary_reference_fits=reference_fits,target_count=len(targets),
    qualified_provenance_count=index["qualified_provenance_count"],
    original_qualified_provenance_count=index["original_qualified_provenance_count"],
    accuracy_requalified_provenance_count=index["accuracy_requalified_provenance_count"],
    excluded_candidate_count=len(excluded),
    exact_deduplication=index["deduplication"],projection_settings=dict(J_mA_cm2=J,Q_mAh_cm2=Q,
      candidate_rule="all independently qualified compatible start/return/best vectors only",
      circuit_starts=6,frequency_points=68,quadrature_orders=[8,16],quadrature_relative_limit=1e-8,
      bias_budget=BUDGET,fit_settings="frozen stage35 extraction-specific setup"),
    spectrum_complex_serialization="[real, imaginary] pairs",raw_circuit_output_dir="circuit_fits_v3",
    no_fit_record_mutation=True)
  save(manifest_path,pm)
  return pm

def verify_projection_freeze():
  pm=read(HERE/"projection_manifest_v3.json");fm_path=HERE/"manifest.json"
  assert sha(fm_path)==pm["fit_manifest_sha256"]
  assert sha(HERE/pm["index_path"])==pm["index_sha256"]
  for rel,digest in pm["fit_record_hashes"].items():assert sha(HERE/rel)==digest,rel
  for rel,digest in pm["source_hashes"].items():assert sha(ROOT/rel)==digest,rel
  source_paths={"run_m_projection_v3.py":HERE/"run_m_projection_v3.py",
    "audit_saved_m_v3.py":HERE/"audit_saved_m_v3.py",
    "run_projection_preflight_v3.py":HERE/"run_projection_preflight_v3.py",
    "qualify_projection_v3.py":HERE/"qualify_projection_v3.py","fit_manifest.json":HERE/"manifest.json",
    "projection_qualification_v3.json":HERE/"projection_qualification_v3.json",
    "controls.json":HERE/"controls.json","sweep35_controls.json":PAPER/"sweep35/controls.json",
    "sweep35_manifest.json":PAPER/"sweep35/manifest.json","m_kernel.py":HERE/"m_kernel.py",
    "sweep35_run_sweep.py":PAPER/"sweep35/run_sweep.py",
    "circuit_fitter.py":PAPER/"impedance8/run_circuit_probe.py",
    "mechanics_waveforms.csv":PAPER/"mechanics3/waveforms.csv",
    "mechanics_waveform_summary.json":PAPER/"mechanics3/waveform_summary.json",
    "accuracy_requalification_manifest.json":HERE/"accuracy_requalification_v1/requalification_manifest.json",
    "accuracy_requalification_targets.json":HERE/"accuracy_requalification_v1/targets.json",
    "accuracy_requalification_audit.json":HERE/"accuracy_requalification_v1/AUDIT.json",
    "accuracy_requalification_execution.json":HERE/"accuracy_requalification_v1/accuracy_all_execution.json"}
  for name,digest in pm["projection_source_hashes"].items():assert sha(source_paths[name])==digest,name
  return pm

def run_target(target,projection_manifest_sha):
  checkpoint_guard();outpath=HERE/target["output_path"]
  if outpath.exists():
    result=read(outpath)
    assert result["projection_manifest_sha256"]==projection_manifest_sha
    return dict(target_id=target["target_id"],reused=True,path=target["output_path"],sha256=sha(outpath),
      numerical_pass=bool(result.get("numerical_pass",False)),projection_failure=result.get("projection_failure"))
  fitpath=HERE/"fits"/(target["canonical_attempt"]+".json");fit=read(fitpath)
  candidate=fit["candidates"][target["canonical_candidate_index"]]
  if target.get("canonical_basis")=="accuracy_requalification_v1":
    accuracy=accuracy_requalification_inputs()
    matches=[r for r in accuracy["rows"] if r["target"]["attempt_id"]==fit["id"] and
      r["target"]["candidate_index"]==target["canonical_candidate_index"]]
    assert len(matches)==1 and candidate["x"]==matches[0]["target"]["x"] and not candidate["qualified_compatible"]
  else:
    assert candidate["qualified_compatible"]
  assert candidate_key(candidate)==target["vector_sha256"]
  try:result=project_candidate(fit["id"],candidate,fit)
  except Exception as exc:
    result=dict(attempt=fit["id"],labels=candidate["labels"],fixed_m=fit["fixed_m"],
      extraction=fit["extraction"],seed_id=fit["seed"]["witness_id"],null=fit["null"],
      x=candidate["x"],projection_failure=type(exc).__name__+": "+str(exc)[:500],numerical_pass=False,
      manifest_sha256=sha(HERE/"manifest.json"),
      within_window_pressure_history_effect="zero by construction" if fit["null"] else "nonzero transfer fit parameter",
      starting_state_difference_retained=True)
  result.update(target_id=target["target_id"],vector_sha256=target["vector_sha256"],
    provenance=target["provenance"],projection_manifest_sha256=projection_manifest_sha,
    canonical_attempt=target["canonical_attempt"])
  save(outpath,result)
  return dict(target_id=target["target_id"],reused=False,path=target["output_path"],sha256=sha(outpath),
    numerical_pass=bool(result.get("numerical_pass",False)),projection_failure=result.get("projection_failure"))

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["freeze","pilot","all"],required=True);a=ap.parse_args()
  checkpoint_guard()
  if a.phase=="freeze":
    pm=freeze_projection_inputs();print(json.dumps({k:v for k,v in pm.items() if k!="source_hashes"},indent=2));return
  pm=verify_projection_freeze();index=read(HERE/pm["index_path"]);targets=index["targets"]
  if a.phase=="pilot":targets=targets[:1]
  if not targets:raise RuntimeError("no qualified compatible fit vectors for projection")
  workers=max(1,int(os.environ.get("AJ_COMPUTE_WORKERS","1")));start=time.perf_counter();done=[]
  pm_sha=sha(HERE/"projection_manifest_v3.json")
  with ProcessPoolExecutor(max_workers=workers) as pool:
    futures=[pool.submit(run_target,t,pm_sha) for t in targets]
    for fut in as_completed(futures):
      result=fut.result();done.append(result)
      save(HERE/"projection_progress_v3.json",dict(phase=a.phase,completed=done,
        planned_targets=len(targets),workers=workers,projection_manifest_sha256=pm_sha))
      print(json.dumps(result),flush=True);checkpoint_guard()
  out=dict(phase=a.phase,target_count=len(targets),completed=done,workers=workers,
      wall_seconds=time.perf_counter()-start,fit_manifest_sha256=sha(HERE/"manifest.json"),
      projection_manifest_sha256=pm_sha)
  save(HERE/(a.phase+"_projection_execution_v3.json"),out)
if __name__=="__main__":main()
