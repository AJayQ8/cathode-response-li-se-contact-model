"""Fixed-m all14 minimax sensitivity fits; explicit m, no historical globals patched."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, os, time
import numpy as np
from scipy.optimize import minimize
from shared_compute import checkpoint_guard

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent
ROOT=HERE.parents[4]
EXPS=(5.9,7.3)
EXTRACTIONS=("ordinary","robust","published")
M6=6.6
SCREEN=.03
MAXITER=60
MAX_FORWARD=200

def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path); v=importlib.util.module_from_spec(sp); sp.loader.exec_module(v); return v
kernel=module("explicit_m_kernel",HERE/"m_kernel.py")
source=kernel.source
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
    if isinstance(v,np.ndarray): return plain(v.tolist())
    if isinstance(v,np.generic): return plain(v.item())
    if isinstance(v,dict): return {str(k):plain(z) for k,z in v.items()}
    if isinstance(v,(list,tuple)): return [plain(z) for z in v]
    if isinstance(v,float) and not math.isfinite(v): return None
    return v
def save(p,v):
    checkpoint_guard();p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+".tmp-"+str(os.getpid()))
    with tmp.open("w") as f: f.write(json.dumps(plain(v),indent=2,allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,p)
def seed_for(e,role):
    return next(s for s in read(HERE/"seed_selection.json")["selected"] if s["extraction"]==e and s["role"]==role)
def task_for(e): return dict(registration="balance_aligned",timing="late",extraction=e)
def protocols_for(e): return source.data(e,"balance_aligned")
def valid_errors(rows):
    if len(rows)!=14 or any(kernel.unavailable(r) for r in rows): return None
    v=np.asarray([r["error"] for r in rows],float)
    return v if np.all(np.isfinite(v)) else None
def in_domain(x,task,rstar,protocols):
    if np.any(x<kernel.LOW) or np.any(x>kernel.HIGH) or np.any(kernel.domain_values(x,rstar,protocols)<-1e-14): return False
    for p in protocols:
        if p["excluded_ambiguous"]: continue
        state,_,a=kernel.initial_state(x,rstar,p["R_ch"])
        if state is None or not 0<a<1 or not source.pressure.domain(p,float(x[2]))["valid"]: return False
    return True

def freeze():
    selection=read(HERE/"seed_selection.json")
    assert selection["pool_count"]==53 and len(selection["selected"])==6
    hashes=dict(selection["source_hashes"])
    paths=[
      HERE/"PLAN.md",HERE/"EXECUTION_APPENDIX.md",HERE/"VERSIONED_REPAIR.md",HERE/"freeze_seeds.py",HERE/"seed_selection.json",
      HERE/"m_kernel.py",Path(__file__),PAPER/"finite22/run_finite_height.py",
      HERE/"manifest_v1.json",HERE/"controls_v1.json",HERE/"run_m_sensitivity_v1.py",
      HERE/"m_kernel_v1.py",HERE/"run_m_projection_v1.py",HERE/"audit_saved_m_v1.py",
      HERE/"job_controls_v2.json",
      PAPER/"pressure21/run_pressure_history.py",PAPER/"geometry20/run_geometry_probe.py",
      PAPER/"robust23/resume_startup_repair.py",PAPER/"decomposition29/manifest.json",
      PAPER/"decomposition29/controls.json",PAPER/"decomposition29/summary.json",
      PAPER/"forecast_split_20260925/manifest.json",PAPER/"forecast_split_20260925/summary.json",
      PAPER/"forecast_split_20260925/post50_extension/manifest.json",
      PAPER/"forecast_split_20260925/post50_extension/summary.json",
      PAPER/"forecast_split_20260925/post50_extension/combined53.json",
      PAPER/"mechanics3/waveforms.csv",PAPER/"mechanics3/waveform_summary.json",
      PAPER/"mechanics3/fullcell_charge_history.json",PAPER/"prospective27/run_projection.py",
      PAPER/"sweep35/run_sweep.py",PAPER/"sweep35/controls.json",PAPER/"sweep35/manifest.json",
      PAPER/"sweep35/PRECISION_REPAIR.md",PAPER/"transfer49/run_prediction.py",
      PAPER/"readout34/run_readout.py",PAPER/"impedance8/run_circuit_probe.py",
      HERE/"run_m_projection.py",HERE/"audit_saved_m.py",
      PAPER/"impedance8/spectra.json"]
    c35=read(PAPER/"sweep35/controls.json")
    assert c35["passed"] and c35["manifest_sha256"]==sha(PAPER/"sweep35/manifest.json")
    for r in c35["records"]:
        assert r["passed"]; paths.append(PAPER/"sweep35"/r["fit_path"])
    for p in paths: hashes[str(p.relative_to(ROOT))]=sha(p)
    for rel,digest in hashes.items(): assert sha(ROOT/rel)==digest,rel
    cfg=dict(schema=2,operational_version=2,seed_selection_sha256=sha(HERE/"seed_selection.json"),sha256=hashes,
      exponents=list(EXPS),m6_reference=M6,extractions=list(EXTRACTIONS),
      registration="balance_aligned",timing="late",seeds=selection["selected"],
      low=kernel.LOW.tolist(),high=kernel.HIGH.tolist(),free_attempts=12,eta_zero_attempts=12,
      eta_zero_exact=True,all14_minimax=True,screen=SCREEN,maxiter=MAXITER,
      max_uncached_forward_evaluations=MAX_FORWARD,null_active=[0,1,3,4],
      forward_solver="LSODA",forward_rtol=1e-9,forward_atol=1e-11,
      independent_solver="Radau",independent_rtol=1e-11,independent_atol=1e-13,
      independent_first_step="min(segment_duration,1e-4)",
      jacobian_steps_normalized=[1e-3,3e-4,1e-4],prediction_limit=1e-7,jacobian_scaled_limit=1e-3,
      minimax_ftol=1e-9,projection_J_mA_cm2=1.2,projection_Q_mAh_cm2=.30,
      spectral_frequencies=68,spectral_order="high_to_low",settle_s=.5,cycles=3,
      quadrature_orders=[8,16],stationary_fit_ftol_xtol_gtol=1e-13,stationary_max_nfev=2500,
      extraction_specific_templates=True,no_kinetic_fit=True,no_bound_scan=True,no_extra_exponents=True,
      endpoint_transform="assign exact stored low/high only when normalized y is exactly 0/1; affine mapping unchanged in interior",
      physical_coordinate_qualification_exponents=list(EXPS),
      physical_coordinate_qualification_states=[dict(label=s["label"],witness=s["witness"],extraction=s["extraction"])
        for s in read(HERE/"controls_v1.json")["qualification_states"]],
      qualification_reuses_controls_v1=True)
    p=HERE/"manifest.json"
    if p.exists() and read(p)!=cfg:
      # Versioned operational repair: v1 receipt/source snapshot is immutable;
      # replace only the active manifest after proving this is the preserved v1.
      assert sha(p)==sha(HERE/"manifest_v1.json"),"Refusing unrecognized active-manifest migration"
      assert not any((HERE/n).exists() for n in ["pilot_execution.json","all_execution.json","fits"])
      save(p,cfg)
    elif not p.exists():
      assert not any((HERE/n).exists() for n in ["controls.json","pilot_execution.json","all_execution.json","fits"])
      save(p,cfg)
    else: assert read(p)==cfg,"Frozen source/selection manifest changed"
    return cfg

def mapping_controls():
    records=[]
    selected=read(HERE/"seed_selection.json")["selected"]
    for seed in selected:
      for null in (False,True):
        p=Minimax(5.9,seed["extraction"],seed,null)
        x=np.asarray(seed["x"],float).copy()
        if null:x[2]=0.
        normalized=(x[p.active]-p.low)/p.width
        y=np.r_[normalized,.1]
        got=p.to_x(y)
        records.append(dict(kind="seed_round_trip",witness=seed["witness_id"],null=null,
          exact=bool(np.array_equal(got,x)),x=x.tolist(),mapped=got.tolist()))
        for value,bound in ((0.,"low"),(1.,"high")):
          yb=np.full(len(p.active)+1,.5);yb[-1]=.1;yb[np.arange(len(p.active))]=value
          xb=p.to_x(yb)
          target=kernel.LOW[p.active] if bound=="low" else kernel.HIGH[p.active]
          records.append(dict(kind="bound_mapping",witness=seed["witness_id"],null=null,
            bound=bound,active=p.active.tolist(),exact=bool(np.array_equal(xb[p.active],target)),
            mapped=xb[p.active].tolist(),target=target.tolist(),eta=xb[2]))
    return dict(name="exact_optimizer_coordinate_mapping",checks=records,
      passed=all(r["exact"] and (not r.get("null",False) or r.get("eta",0.)==0.) for r in records))

def endpoint_physical_controls():
    old=read(HERE/"controls_v1.json")
    assert old["passed"] and old["manifest_sha256"]==sha(HERE/"manifest_v1.json")
    tests=[];selected=read(HERE/"seed_selection.json")["selected"]
    chosen={s["witness_id"]:s for s in selected}
    for state in old["qualification_states"]:
      seed=chosen[state["witness"]]
      assert seed["extraction"]==state["extraction"] and seed["x"]==state["x"]
      e=seed["extraction"];x=np.asarray(seed["x"],float);task=task_for(e);rstar,protocols=protocols_for(e)
      for m in EXPS:
        rows=kernel.predictions_m(x,task,rstar,protocols,m,True)
        available=valid_errors(rows) is not None
        fast_physical=available and all(kernel.physical_row(r) for r in rows)
        domain=in_domain(x,task,rstar,protocols)
        independent=kernel.physical_coordinate_audit(x,task,rstar,protocols,m,rows)
        key=lambda r:(r["cathode"],r["rate_c"],r["stage"],r["subset"])
        row_order=(len(rows)==14 and [key(r) for r in rows]==[key(r) for r in independent.get("rows",[])])
        event_match=(row_order and all(bool(r.get("disconnected",False))==bool(q.get("independent_stopped",False))
                                       for r,q in zip(rows,independent["rows"])))
        maxima=independent.get("maxima",{})
        ratio=maxima.get("ratio_difference",float("inf"))
        passed=bool(available and fast_physical and domain and independent.get("passed",False)
                    and row_order and event_match and ratio<=1e-7)
        tests.append(dict(name=f"independent_physical_route_{state['label']}_m{m}",
          witness=seed["witness_id"],extraction=e,seed=seed["role"],m=m,x=x.tolist(),
          available=available,fast_physical_pass=bool(fast_physical),domain_pass=bool(domain),
          row_order_match=row_order,event_domain_flags_match=event_match,
          independent=independent,passed=passed))
        checkpoint_guard()
    return tests

def qualification_controls():
    path=HERE/"controls.json"
    if path.exists() and read(path).get("manifest_sha256")==sha(HERE/"manifest.json"):
      prior=read(path)
      assert prior["passed"],"Preserved qualification failure"
      return prior
    v1=read(HERE/"controls_v1.json")
    assert v1["passed"] and v1["manifest_sha256"]==sha(HERE/"manifest_v1.json"),"Preserved qualification failure"
    tests=list(v1["tests"])
    # The original m=6.6/Jacobian/physical controls are reused byte-for-byte from v1.
    # Only add the exact bound map repair and endpoint-m independent physical routes.
    mapping=mapping_controls();tests.append(mapping)
    tests.extend(endpoint_physical_controls())
    report=dict(tests=tests,qualification_states=v1["qualification_states"],
               passed=all(t["passed"] for t in tests),manifest_sha256=sha(HERE/"manifest.json"),
               previous_v1_controls_sha256=sha(HERE/"controls_v1.json"),version=2)
    save(path,report)
    assert report["passed"],"Frozen version-2 m/physical/mapping qualification failed; preserve receipt"
    return report

class Minimax:
  def __init__(self,m,e,seed,null):
    self.m,self.e,self.seed,self.null=m,e,seed,null;self.task=task_for(e);self.rstar,self.protocols=protocols_for(e)
    self.active=np.array([0,1,3,4] if null else [0,1,2,3,4]);self.low=kernel.LOW[self.active];self.width=kernel.WIDTH[self.active]
    self.last_x=None;self.r=self.j=None;self.evals=0;self.after_cap=0;self.capped=False;self.failures={};self.best=None;self.trace=[]
  def to_x(self,y):
    x=np.asarray(self.seed["x"],float).copy()
    if self.null:x[2]=0.
    u=np.asarray(y[:len(self.active)],float)
    mapped=self.low+self.width*u
    mapped=np.where(u==0.,self.low,mapped)
    mapped=np.where(u==1.,kernel.HIGH[self.active],mapped)
    x[self.active]=mapped
    return x
  def calc(self,y):
    x=self.to_x(y)
    if self.last_x is not None and np.array_equal(x,self.last_x):return
    if self.evals>=MAX_FORWARD:
      self.capped=True;self.after_cap+=1;self.last_x=x;self.r=np.full(14,100.);self.j=np.zeros((14,len(self.active)));return
    self.evals+=1
    if self.evals%10==0:checkpoint_guard()
    try:
      rows=kernel.predictions_m(x,self.task,self.rstar,self.protocols,self.m,True)
      if len(rows)!=14:raise RuntimeError(f"expected14 got{len(rows)}")
      r=np.asarray([100. if kernel.unavailable(a) else a["error"] for a in rows],float)
      j=np.asarray([a["derivative"] for a in rows],float)[:,self.active]
      if not np.all(np.isfinite(r)) or not np.all(np.isfinite(j)):raise FloatingPointError("nonfinite fit")
      maxerr=float(np.max(np.abs(r)))
      if all(not kernel.unavailable(a) and kernel.physical_row(a) for a in rows) and in_domain(x,self.task,self.rstar,self.protocols):
        if self.best is None or maxerr<self.best["max_abs_error"]:
          self.best=dict(x=x.tolist(),max_abs_error=maxerr,all14_SSE=float(r@r),evaluation=self.evals);self.trace.append(self.best.copy())
    except (ValueError,RuntimeError,FloatingPointError,OverflowError) as exc:
      msg=type(exc).__name__+": "+str(exc)[:180];self.failures[msg]=self.failures.get(msg,0)+1;r=np.full(14,100.);j=np.zeros((14,len(self.active)))
    self.last_x,self.r,self.j=x,r,j
  def domain(self,y): return kernel.domain_values(self.to_x(y),self.rstar,self.protocols)
  def domain_jac(self,y):
    b,A=self.to_x(y)[3:];orig=np.array([0.,0.,0.,A-1.,b-1.])
    return np.tile(orig[self.active]*self.width,(len(self.domain(y)),1))
  def constr(self,y):
    self.calc(y);return np.r_[y[-1]-self.r,y[-1]+self.r,self.domain(y)]
  def constr_jac(self,y):
    self.calc(y);j=self.j*self.width
    return np.r_[np.c_[-j,np.ones(14)],np.c_[j,np.ones(14)],np.c_[self.domain_jac(y),np.zeros(len(self.domain(y)))]]
  def start(self):
    x=np.asarray(self.seed["x"],float).copy()
    if self.null:x[2]=0.
    y=np.r_[(x[self.active]-self.low)/self.width,.1];self.calc(y);y[-1]=min(100.,max(0.,np.max(np.abs(self.r))+1e-6));return y

def fit_id(m,e,seed,null):return f"m{str(m).replace('.','p')}_{e}_{seed['witness_id']}_{'eta0' if null else 'free'}"
def assess(m,e,seed,x,labels):
  task=task_for(e);rstar,protocols=protocols_for(e);x=np.asarray(x,float)
  try:
    rows=kernel.predictions_m(x,task,rstar,protocols,m,True);r=valid_errors(rows);available=r is not None
    physical_fast=available and all(kernel.physical_row(v) for v in rows);domain=in_domain(x,task,rstar,protocols)
    maxerr=None if r is None else float(np.max(np.abs(r)));sse=None if r is None else float(r@r)
    audit=kernel.physical_coordinate_audit(x,task,rstar,protocols,m,rows)
    disagreement=audit.get("maxima",{}).get("ratio_difference",math.inf)
    compatible=bool(available and physical_fast and domain and audit["passed"] and maxerr<=SCREEN and SCREEN-maxerr>disagreement)
    return dict(labels=labels,x=x.tolist(),all14_rows=rows,all14_available=available,max_abs_error=maxerr,all14_SSE=sse,
      physical_fast_pass=bool(physical_fast),domain_pass=bool(domain),physical_coordinate_audit=audit,
      numeric_disagreement=disagreement,screen=SCREEN,screen_margin=None if maxerr is None else SCREEN-maxerr,
      qualified_compatible=compatible)
  except (ValueError,RuntimeError,FloatingPointError,OverflowError,AssertionError) as exc:
    return dict(labels=labels,x=x.tolist(),all14_rows=[],all14_available=False,max_abs_error=None,all14_SSE=None,
      physical_fast_pass=False,domain_pass=False,physical_coordinate_audit=dict(passed=False,error=type(exc).__name__+": "+str(exc)[:250]),
      numeric_disagreement=None,screen=SCREEN,screen_margin=None,qualified_compatible=False)

def run_attempt(spec):
  tic=time.perf_counter()
  m,e,seed,null=spec;ident=fit_id(m,e,seed,null);path=HERE/"fits"/(ident+".json")
  if path.exists():
    old=read(path);assert old["manifest_sha256"]==sha(HERE/"manifest.json") and old["seed"]==seed
    return old
  p=Minimax(m,e,seed,null);y0=p.start();n=len(p.active)
  try:
    opt=minimize(lambda y:float(y[-1]),y0,jac=lambda y:np.r_[np.zeros(n),1.],method="SLSQP",
       bounds=[(0.,1.)]*n+[(0.,100.)],constraints=[dict(type="ineq",fun=p.constr,jac=p.constr_jac)],
       options=dict(maxiter=MAXITER,ftol=1e-9,disp=False))
    optrec=dict(success=bool(opt.success),status=int(opt.status),message=str(opt.message),nit=int(opt.nit),nfev=int(opt.nfev),
       njev=int(opt.njev),epigraph=float(opt.fun),x=p.to_x(opt.x).tolist(),y=np.asarray(opt.x).tolist(),
       minimum_constraint=float(np.min(p.constr(opt.x))))
  except Exception as exc: opt=None;optrec=dict(success=False,status="exception",message=type(exc).__name__+": "+str(exc)[:250],x=None)
  candidates=[("start",np.asarray(seed["x"],float).copy())]
  if null:candidates[0][1][2]=0.
  if optrec["x"] is not None:candidates.append(("optimizer_return",np.asarray(optrec["x"],float)))
  if p.best is not None:candidates.append(("best_visited",np.asarray(p.best["x"],float)))
  unique={}
  for label,x in candidates:
    key=tuple(float(v) for v in x)
    if key not in unique:unique[key]=dict(labels=[label],x=x)
    else:unique[key]["labels"].append(label)
  checked=[assess(m,e,seed,v["x"],v["labels"]) for v in unique.values()]
  result=dict(id=ident,fixed_m=m,extraction=e,setting=task_for(e),null=bool(null),eta_constraint=0. if null else None,
     seed=seed,optimizer=optrec,optimizer_caps=dict(maxiter=MAXITER,max_uncached_forward_evaluations=MAX_FORWARD),
     optimizer_forward_evaluations=p.evals,optimizer_forward_cap_reached=p.capped,post_cap_optimizer_requests=p.after_cap,
     evaluation_failures=p.failures,best_visited=p.best,improvement_trace=p.trace,candidates=checked,
     wall_seconds=None,manifest_sha256=sha(HERE/"manifest.json"),projections_completed=False)
  result["wall_seconds"]=float(time.perf_counter()-tic)
  save(path,result);return result

def tasks():
  selected=read(HERE/"seed_selection.json")["selected"];out=[]
  for m in EXPS:
    for e in EXTRACTIONS:
      for s in [v for v in selected if v["extraction"]==e]:
        out.extend([(m,e,s,False),(m,e,s,True)])
  return out

def execute(phase):
  frozen=freeze();controls=qualification_controls()
  if phase=="controls":return dict(phase=phase,planned=0,completed=[],controls_passed=True)
  work=tasks()
  if phase=="pilot":
    work=[t for t in work if t[0]==5.9 and t[1]=="ordinary" and t[2]["role"]=="minimum_all14_SSE" and not t[3]]
  elif phase=="all":
    work=[t for t in work if not ((HERE/"fits"/(fit_id(*t)+".json")).exists() and
             read(HERE/"fits"/(fit_id(*t)+".json")).get("projections_completed"))]
  workers=int(os.environ.get("AJ_COMPUTE_WORKERS","1"));started=time.perf_counter();completed=[]
  with ProcessPoolExecutor(max_workers=workers) as pool:
    for fut in as_completed([pool.submit(run_attempt,t) for t in work]):
      r=fut.result();item=dict(id=r["id"],m=r["fixed_m"],extraction=r["extraction"],null=r["null"],
        candidates=len(r["candidates"]),qualified=sum(c["qualified_compatible"] for c in r["candidates"]),
        optimizer_success=r["optimizer"].get("success",False),cap=r["optimizer_forward_cap_reached"],seconds=r["wall_seconds"])
      completed.append(item);print(json.dumps(item),flush=True);checkpoint_guard()
  out=dict(phase=phase,planned=len(work),completed=completed,workers=workers,wall_seconds=time.perf_counter()-started,
     manifest_sha256=sha(HERE/"manifest.json"),controls_passed=controls["passed"],source_count=len(frozen["sha256"]))
  save(HERE/(phase+"_execution.json"),out);return out

started=time.perf_counter()
def main():
  a=argparse.ArgumentParser();a.add_argument("--phase",choices=["controls","pilot","all"],required=True);args=a.parse_args()
  checkpoint_guard();result=execute(args.phase);print(json.dumps(dict(phase=args.phase,planned=result["planned"],
    wall_seconds=result.get("wall_seconds",0.),passed=True)),flush=True)
if __name__=="__main__":main()
