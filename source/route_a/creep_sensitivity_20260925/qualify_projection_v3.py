"""Static inheritance qualification for the requalified-vector projection extension."""
from pathlib import Path
import ast
import hashlib
import json

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def function_tree(path,name):
  tree=ast.parse(Path(path).read_text())
  node=next(item for item in tree.body if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)) and item.name==name)
  return node

def normalize_v3_to_v2(node):
  class Normalize(ast.NodeTransformer):
    def visit_Constant(self,n):
      if isinstance(n.value,str):n.value=n.value.replace("_v3","_v2").replace("v3","v2")
      return n
  return ast.dump(Normalize().visit(ast.fix_missing_locations(node)),include_attributes=False)

def main():
  old=read(HERE/"projection_qualification_v2.json")
  assert old["passed"] and old["checks"]["passed"]
  checks={}
  for name in ("project_candidate",):
    before=function_tree(HERE/"run_m_projection_v2.py",name)
    after=function_tree(HERE/"run_m_projection_v3.py",name)
    checks[name+"_unchanged_after_version_path_normalization"]=normalize_v3_to_v2(before)==normalize_v3_to_v2(after)
  for name in ("audit_cell","check_fit_inputs"):
    before=function_tree(HERE/"audit_saved_m_v2.py",name)
    after=function_tree(HERE/"audit_saved_m_v3.py",name)
    checks[name+"_unchanged"]=ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
  for name,value in checks.items():assert value,name

  accuracy_dir=HERE/"accuracy_requalification_v1"
  accuracy=read(accuracy_dir/"AUDIT.json")
  accuracy_targets=read(accuracy_dir/"targets.json")
  accuracy_exec=read(accuracy_dir/"accuracy_all_execution.json")
  assert accuracy["passed"] and accuracy["complete_four_target_count"]==4 and not accuracy["failures"]
  assert accuracy["target_index_sha256"]==sha(accuracy_dir/"targets.json")
  assert accuracy["requalification_manifest_sha256"]==sha(accuracy_dir/"requalification_manifest.json")
  assert accuracy_exec["phase"]=="all" and accuracy_exec["target_count"]==4
  assert len(accuracy_targets["targets"])==4
  for relative,digest in accuracy["result_hashes"].items():assert sha(accuracy_dir/relative)==digest

  out=dict(schema=1,passed=True,purpose="Reuse the already-qualified fixed-m projection physics and circuit extractor while adding four independently audited exact-vector requalifications; no new numerical integrations in this qualification.",
    inherited_projection_qualification_sha256=sha(HERE/"projection_qualification_v2.json"),
    projection_script_sha256=sha(HERE/"run_m_projection_v3.py"),
    audit_script_sha256=sha(HERE/"audit_saved_m_v3.py"),
    preflight_script_sha256=sha(HERE/"run_projection_preflight_v3.py"),
    qualification_script_sha256=sha(Path(__file__)),
    v2_projection_script_sha256=sha(HERE/"run_m_projection_v2.py"),
    v2_audit_script_sha256=sha(HERE/"audit_saved_m_v2.py"),
    accuracy_requalification=dict(audit_sha256=sha(accuracy_dir/"AUDIT.json"),
      manifest_sha256=sha(accuracy_dir/"requalification_manifest.json"),
      target_index_sha256=sha(accuracy_dir/"targets.json"),execution_sha256=sha(accuracy_dir/"accuracy_all_execution.json"),
      result_count=len(accuracy["result_hashes"]),all_four_newly_qualified=True),
    checks=checks,
    inherited_stationary_and_six_start_tests=old["checks"],
    scope="Static inheritance only: v2 independently qualified the unchanged physical projector, extraction-specific stationary references and six-start circuit objective. This version adds no new model or circuit setup. Four vectors are admitted only via separate exact-vector two-level accuracy receipts; the original failed fit records remain unchanged.")
  Path(HERE/"projection_qualification_v3.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
  print(json.dumps({"passed":out["passed"],"checks":checks,
    "qualification_sha256":sha(HERE/"projection_qualification_v3.json")},indent=2))

if __name__=="__main__":main()
