from __future__ import annotations
import gzip, hashlib, json, resource, sys, time
import argparse
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

RESEARCH: Path
BASE: Path
PROTOCOL: Path
RUNTIME: Path
from pess_ess import solve
from pess_ess.model import Candidate, Instance, Task


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def expected(dataset: str, objective: str, budget: int):
    refs=json.loads(PROTOCOL.read_text())['frozen_exact_references'][dataset][objective][str(budget)]
    return Fraction(refs['objective_fraction']), tuple(tuple(x) for x in refs['all_optimal_sets'])

def forest():
    root=BASE/'generalization-issue64'
    frozen_path=root/'forest10-valid-frozen-selection-v1.json'
    if sha(frozen_path)!='10bcf1f6026aa4de025cbef360c8302aafb2f082d49138c4c5da2d1331f4c798':
        raise ValueError('Forest10 frozen selection hash mismatch')
    frozen=json.loads(frozen_path.read_text())
    ids=tuple(frozen['inputs']['candidate_order'])
    if len(ids)!=45 or frozen['inputs']['valid_workload_count']!=10000: raise ValueError('Forest10 frozen dimensions mismatch')
    coeff=root/'coefficients.jsonl.gz'
    if sha(coeff)!='b1f0e6521804cc0e77f0616271d94b10a3602bde6eb677591ee040e14c7dbd04': raise ValueError('Forest10 coefficient hash mismatch')
    candidates=tuple(Candidate(e,'forest10',(e[1:3],e[4:6]),Fraction(1)) for e in ids)
    tasks=[]
    digest=hashlib.sha256()
    with gzip.open(coeff,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); b={k:Fraction(v) for k,v in r['b_fraction'].items()}
            model={'alpha':r['alpha_fraction'],'b':{k:str(v) for k,v in sorted(b.items())},'available_edges':tuple(b)}
            digest.update((json.dumps(model,sort_keys=True,separators=(',',':'))+'\n').encode())
            tasks.append(Task(r['query_id'],'forest10',Fraction(1),Fraction(r['alpha_fraction']),tuple(b),b))
    if len(tasks)!=10000: raise ValueError('Forest10 task count mismatch')
    return ids,candidates,tuple(tasks),digest.hexdigest(),{'coefficients_sha256':sha(coeff),'frozen_sha256':sha(frozen_path)},ids

def census():
    root=BASE/'census13-issue71'
    run=RUNTIME/'ess-research/issue71/runs/20261022-084335-census13-full-valid'
    plan=json.loads((root/'census13-stage2b-plan-v1.json').read_text())
    frozen_path=root/'census13-clamped-log-v2-schema-order-frozen-selection-v1.json'
    plan_path=root/'census13-stage2b-plan-v1.json'
    expected_inputs={
        'plan_sha256':'51842c6981936d9a5be39342eb5caee23b35f3137e5876064b0b1a649906d5e1',
        'frozen_sha256':'7c4e0abeaff84a147c80a77a6d854df12950600cf2139a4fc661c21e140ff7a7',
        'valid_truth_sha256':'2b88019f55b3e1ba172c5851e8c6adfb0f37ad878386e244e83d6fa3046ba28d',
        'worker0_sha256':'3dccb9675648f1475bb098ecc8288b2b61bf8f4c13cdaee87a6ca276c8937602',
        'worker1_sha256':'341cebd27650cf34608073a4712b71fd1c4a26cc6338d581c1d1e01117ef88e4',
        'coverage_sha256':'ce33ae7a290686b18b9ac14ca628b24827210257b501bedc2ddf49408ac14f7d',
    }
    actual_inputs={
        'plan_sha256':sha(plan_path),
        'frozen_sha256':sha(frozen_path),
        'valid_truth_sha256':sha(run/'valid-truth-counts.jsonl'),
        'worker0_sha256':sha(run/'observations/worker0.jsonl'),
        'worker1_sha256':sha(run/'observations/worker1.jsonl'),
        'coverage_sha256':sha(run/'coverage.jsonl'),
    }
    if actual_inputs != expected_inputs:
        raise ValueError('Census13 frozen model input hash mismatch')
    frozen=json.loads(frozen_path.read_text())
    verified=json.loads((run/'verification.json').read_text())
    if verified.get('status')!='PASS' or verified.get('total_calls')!=288849 or verified.get('valid_query_count')!=10000:
        raise ValueError('Census13 full-valid observation verification mismatch')
    candidates=tuple(Candidate(x['id'],'census13',tuple(x['endpoints']),Fraction(1)) for x in plan['candidate_policy']['candidates'])
    ids=tuple(x.id for x in candidates)
    if len(ids)!=78 or tuple(frozen['candidate_ids'])!=ids: raise ValueError('Census13 candidate order mismatch')
    truth={};
    with (run/'valid-truth-counts.jsonl').open() as f:
        for line in f:
            r=json.loads(line); truth[r['query_id']]=r['true_cardinality']
    empty={}; singleton={}
    for worker in (0,1):
        with (run/'observations'/f'worker{worker}.jsonl').open() as f:
            for line in f:
                r=json.loads(line)
                if r.get('status')!='SUCCESS': raise ValueError('non-success frozen Census observation')
                q,c=r['query_id'],r['config_id']; v=Decimal(r['nrows_preclamp'])
                if c=='Empty': empty[q]=max(Decimal(1),v)
                else: singleton.setdefault(c,{})[q]=max(Decimal(1),v)
    available={}
    with (run/'coverage.jsonl').open() as f:
        for line in f:
            r=json.loads(line); available[r['query_id']]=tuple(r['eligible_config_ids'])
    if len(empty)!=10000 or len(singleton)!=78 or len(available)!=10000 or set(truth)!=set(empty):
        raise ValueError('Census13 model input coverage mismatch')
    helper=run/'pinned-helpers'; sys.path.insert(0,str(helper))
    import forest10_runner
    query_ids=tuple(truth)
    models=forest10_runner.build_task_models(query_ids,truth,empty,singleton,available)
    digest=hashlib.sha256('\n'.join(json.dumps({'alpha':str(m.alpha),'b':{k:str(v) for k,v in sorted(m.b.items())},'available_edges':m.available_edges},sort_keys=True,separators=(',',':')) for m in models).encode()).hexdigest()
    if digest!='7671c4d8837c4f63423bb4a0cf1a132923301ef4f6d04cdf5ee380dfe6cbc0a5': raise ValueError('Census13 model coefficient digest mismatch')
    tasks=tuple(Task(q,'census13',Fraction(1),m.alpha,m.available_edges,m.b) for q,m in zip(query_ids,models))
    order=tuple(plan['nominal_priority_orders']['orders'][0]['candidate_ids'])
    return ids,candidates,tasks,digest,actual_inputs | {'model_digest':digest},order

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("forest10", "census13"))
    parser.add_argument("objective", choices=("nominal", "pess"))
    parser.add_argument("budget", type=int, choices=(2, 3))
    parser.add_argument("output", type=Path)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--runtime-parent", type=Path, required=True)
    args = parser.parse_args()
    global RESEARCH, BASE, PROTOCOL, RUNTIME
    RESEARCH = args.research_root
    BASE = RESEARCH / 'experiments/pess-multidataset-heldout-stage2b'
    PROTOCOL = RESEARCH / 'experiments/solver-integration-issue74/stage2-b23-protocol-v1.json'
    RUNTIME = args.runtime_parent
    dataset, objective, budget, out = args.dataset, args.objective, args.budget, args.output
    start=time.monotonic()
    ids,candidates,tasks,model_digest,hashes,order=(forest() if dataset=='forest10' else census())
    instance=Instance(candidates,tasks,Fraction(budget))
    result=solve(instance,objective=objective,backend='exact_enumeration',priority_order=order if objective=='nominal' else None,total_time_limit=895)
    ref_value,ref_sets=expected(dataset,objective,budget)
    found=tuple(tuple(sorted(x)) for x in result.optimal_sets)
    matches=result.complete and result.objective==ref_value and found==ref_sets
    report={'case_id':f'{dataset}-{objective}-b{budget}','dataset':dataset,'objective':objective,'budget':budget,'task_count':len(tasks),'candidate_count':len(ids),'model_digest':model_digest,'input_hashes':hashes,'complete':result.complete,'status':'PASS' if matches else ('INTERRUPTED' if not result.complete else 'REFERENCE_MISMATCH'),'objective_fraction':str(result.objective) if result.objective is not None else None,'reference_objective_fraction':str(ref_value),'optimal_sets':[list(x) for x in found], 'reference_optimal_sets':[list(x) for x in ref_sets],'evaluated_subsets':result.evaluated_subsets,'feasible_subsets':result.feasible_subsets,'pruned_branches':result.pruned_branches,'backend_elapsed_seconds':result.elapsed_seconds,'process_elapsed_seconds':time.monotonic()-start,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,sort_keys=True),flush=True)
    return 0 if matches else 3
if __name__=='__main__': raise SystemExit(main())
