"""Check diagonal forecast inputs for exact identity; no numerical work."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
REV=HERE.parent


def read(p):
    return json.loads(p.read_text())


def main():
    manifest=read(HERE/'manifest.json')
    count=0
    for w in manifest['witnesses']:
        source=read(REV/'sweep35/cases'/(w['id']+'_fixed_q_high_to_low.json'))
        rest=read(REV/'readout34/cases'/(w['id']+'.json'))
        for initial in ['P-LCO','N-LCO']:
            diagonal=next(h for h in w['diagonal_histories'] if h['cathode']==initial)
            cell=next(h for h in source['histories'] if h['cathode']==initial)
            state=next(h for h in rest['histories'] if h['cathode']==initial)
            assert cell['template']['Rref']==state['R_ch']==diagonal['R_ch']
            assert state['q0']==diagonal['q0']
            assert rest['witness']==w
            count+=1
    assert count==100
    output=dict(passed=True,checked_diagonal_input_identities=count,
                checked=['witness','initial R_ch','reference template Rref','history q0'],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                numerical_recalculations=0)
    (HERE/'diagonal_input_check.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))


if __name__=='__main__':main()
