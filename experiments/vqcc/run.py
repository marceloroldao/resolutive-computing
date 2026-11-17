"""Reproduce full-domain screening and native verification on GitHub Actions."""
import dataclasses
import hashlib
import importlib
import json
import math
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(os.environ.get('VQCC_OUTPUT', 'vqcc-results')).resolve()
VENDOR = OUT / 'upstream'
N = 3133452
MIRRORS = [
 ('Inzainos/quantum-circuit-compilation','4277c767440df9efb8a90860ae2ce47028e46ba9'),
 ('nhphan0505/verifiable-quantum-circuit','872bb089cbf62931fd156f5c86243c7bd053c9a0'),
]
FILES = {
 'nr_reference_circuit.py':'4fb63a258e1eea2f35312f77785902f9bcc914d5',
 'verify_submission_invsqrt.py':'1fa529e9790a5cb000ee7f084e197c9252e5f9d7',
}
def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')

def fetch():
    VENDOR.mkdir(parents=True, exist_ok=True)
    receipt = []
    for name, expected in FILES.items():
        downloaded=[]
        for repo,sha in MIRRORS:
            url=f'https://raw.githubusercontent.com/{repo}/{sha}/verifiers/invsqrt/{name}'
            with urllib.request.urlopen(url, timeout=60) as response:
                data=response.read()
            blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if blob != expected:
                raise RuntimeError(f'Upstream blob mismatch: {name}')
            downloaded.append(data)
            receipt.append(dict(url=url,git_blob_sha1=blob,sha256=hashlib.sha256(data).hexdigest()))
        if downloaded[0]!=downloaded[1]:
            raise RuntimeError('Public mirrors disagree')
        (VENDOR/name).write_bytes(downloaded[0])
    repo,sha=MIRRORS[0]
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/{repo}/{sha}/verifiers/LICENSE.txt', timeout=60) as response:
        (VENDOR/'LICENSE.txt').write_bytes(response.read())
    dump(OUT/'provenance.json', dict(
        source='Competition package mirrored publicly; identical in two pinned repositories',
        authenticated_kaggle_download=False, files=receipt))
    sys.path.insert(0,str(VENDOR))
    sys.path.insert(0,str(ROOT))

def screen(prec, candidate, verifier):
    import numpy as np
    spec=verifier.ProblemSpecInvSqrt.for_n_p(10)
    worst=0.0
    worst_x=0
    failures=0
    start=time.monotonic()
    for lo in range(1,N+1,65536):
        xs=np.arange(lo,min(N+1,lo+65536),dtype=np.int64)
        ys=candidate.model(prec,xs)
        target,tol=verifier.reference_values(spec,xs.astype(np.uint64))
        phase=ys.astype(np.uint64) << np.uint64(64-prec.output_bits(10))
        with np.errstate(over='ignore'):
            error=(phase-target).view(np.int64)
        used=np.abs(error).view(np.uint64)
        failures+=int((used>tol).sum())
        ratios=used.astype(float)/tol.astype(float)
        index=int(np.argmax(ratios))
        if ratios[index]>worst:
            worst=float(ratios[index])
            worst_x=int(xs[index])
    return dict(precision=dataclasses.asdict(prec),inputs=N,failures=failures,
                worst_error_over_tolerance=worst,worst_x=worst_x,seconds=time.monotonic()-start,
                method='exact integer column-truncation model, uint64 phase reference')

def verify_file(name, path):
    args=[sys.executable,str(VENDOR/'verify_submission_invsqrt.py'),str(path),
          '--n-p','10','--beta','1','--precision-bits','24','--eta','218',
          '--max-x',str(N),'--shots-per-x','64','--batch-size','256',
          '--max-workers','2','--salt','resolutive-vqcc-experiment-v1']
    start=time.monotonic()
    proc=subprocess.run(args,text=True,capture_output=True,timeout=1500)
    (OUT/f'{name}.stdout.log').write_text(proc.stdout)
    (OUT/f'{name}.stderr.log').write_text(proc.stderr)
    try:
        result=json.loads(proc.stdout)
    except json.JSONDecodeError:
        result=dict(status='ERROR',error='Verifier did not return JSON',returncode=proc.returncode)
    result['wall_seconds']=time.monotonic()-start
    result['command']=args
    result['returncode']=proc.returncode
    if result.get('status')=='PASS' and result.get('verified_shots')!=N*64:
        raise RuntimeError('Refusing to label reduced verification as full verification')
    dump(OUT/f'{name}.json',result)
    print(name,json.dumps(result),flush=True)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fetch()
    import numpy as np
    import kickmix as km
    import nr_reference_circuit as ref
    import verify_submission_invsqrt as verifier
    import candidate
    report=dict(
        status='RUNNING',physics_validated=False,ro_optimizer_tested=False,
        upstream_provenance='pinned public mirrors, not authenticated against Kaggle',
        native_kickmix=str(getattr(km,'__version__','unknown')),
        full_inputs=N,shots_per_input=64,screenings=[])
    dump(OUT/'report.json',report)
    # Negative control: a no-op circuit must be rejected because it has no target phase.
    builder=km.CircuitBuilder()
    builder.create_quantum_register(22,name='x')
    empty=builder.finish_circuit()
    tiny=dataclasses.replace(verifier.ProblemSpecInvSqrt.for_n_p(10),max_x=64)
    try:
        verifier.verify(empty,tiny,max_workers=1,salt='negative-control')
    except AssertionError as ex:
        report['negative_control']=dict(status='REJECTED_AS_EXPECTED',error=str(ex))
    else:
        raise RuntimeError('Verifier accepted a no-phase circuit')

    configs=[
        ref.Precision(14,17,28,28,29),
        ref.Precision(16,18,30,30,31),
        ref.Precision(18,20,32,32,33),
    ]
    eligible=[]
    for i,prec in enumerate(configs):
        result=screen(prec,candidate,verifier)
        result['configuration']=i
        if result['failures']==0 and result['worst_error_over_tolerance']<0.95:
            circuit=candidate.build(prec)
            stats=circuit.operation_statistics()
            path=OUT/f'candidate-{i}.kmb'
            circuit.to_file(str(path),format='kmb')
            result.update(num_qubits=int(circuit.num_qubits),
                          toffoli_count=int(stats.num_toffoli_like),
                          selection_proxy=int(circuit.num_qubits)*int(stats.num_toffoli_like))
            eligible.append((result['selection_proxy'],path,i))
        report['screenings'].append(result)
        dump(OUT/'report.json',report)
        print('screen',json.dumps(result),flush=True)

    baseline=ref.build_nr_circuit(10)
    baseline_path=OUT/'reference.kmb'
    baseline.to_file(str(baseline_path),format='kmb')
    report['reference']=verify_file('reference',baseline_path)
    dump(OUT/'report.json',report)
    if not eligible:
        report['status']='NO_VALID_CANDIDATE'
    else:
        _,path,i=min(eligible)
        report['selected_configuration']=i
        report['candidate']=verify_file('candidate',path)
        if report['reference']['status']=='PASS' and report['candidate']['status']=='PASS':
            ratio=report['reference']['score']/report['candidate']['score']
            report.update(status='FULL_LOCAL_VERIFICATION_PASS',improvement_vs_reference=ratio,
                          lower_cost_than_reference=ratio>1,
                          prize_probability='not established; no leaderboard comparison')
        else:
            report['status']='NATIVE_VERIFICATION_FAILED'
    dump(OUT/'report.json',report)
    summary='# VQCC inverse-square-root experiment\n\n'
    summary+='Status: '+report['status']+'\n\n'
    for name in ('reference','candidate'):
        r=report.get(name,{})
        summary+=f"- {name}: status={r.get('status')}, Q={r.get('num_qubits')}, C={r.get('toffoli_equivalent')}, Q*C={r.get('score')}, shots={r.get('verified_shots')}\n"
    summary+='\nImprovement vs reference: '+str(report.get('improvement_vs_reference','not established'))+'\n'
    summary+='\nThis tests conventional circuit optimization, not new physics or RO superiority. Verifier provenance is public mirrors, pending authenticated Kaggle comparison.\n'
    (OUT/'SUMMARY.md').write_text(summary)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:
            f.write(summary)
    print(json.dumps(report,indent=2),flush=True)
    if report['status']!='FULL_LOCAL_VERIFICATION_PASS':
        raise SystemExit(1)

if __name__=='__main__':
    main()
