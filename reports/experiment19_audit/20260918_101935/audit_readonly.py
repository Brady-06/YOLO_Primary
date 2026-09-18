import os,sys,pathlib,json,csv,io,hashlib,contextlib,copy,importlib.metadata as im
os.chdir('/root/YOLO'); sys.path.insert(0,'/root/YOLO/scripts')
import torch,yaml
from ultralytics import YOLO
import torch_pruning as tp
from thop import profile
torch.set_num_threads(2)
root=pathlib.Path('/root/YOLO')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
result={'environment':{k:im.version(k) for k in ['torch','ultralytics','torch-pruning','ultralytics-thop']},'checkpoints':{},'splits':{},'text_files':{}}
weights={str(root/'weights/yolo11s.pt')}
for p in root.glob('runs/recovery/experiment19[b-c]_coco2017/*/*/comparison.csv'):
    d=json.loads((p.parent/'run_info.json').read_text())
    if d['configuration']['epochs']!=20: continue
    for row in csv.DictReader(io.StringIO(p.read_text())):
        if row['checkpoint']=='source' or row['selected'].lower()=='true':weights.add(row['weights'])
    for f in ['run_info.json','training/args.yaml','training/results.csv']:
        fp=p.parent/f
        if fp.exists() and f.endswith('results.csv'):
            rows=list(csv.DictReader(io.StringIO(fp.read_text())))
            result.setdefault('training',{})[str(p.parent.relative_to(root))]={'rows':len(rows),'last':rows[-1] if rows else None}
# inspect partition lists and their intersections (resolved filenames)
splitroot=root/'runs/analysis/experiment16_taylor_sensitivity_coco2017/20260915_122417'
for p in splitroot.glob('*.yaml'):
    result['text_files'][str(p.relative_to(root))]=p.read_text()
sets={}
for name,key in [('calibration','train'),('tune','val'),('recovery','train')]:
    p=splitroot/(name+'.yaml')
    if not p.exists():continue
    cfg=yaml.safe_load(p.read_text()); v=cfg.get(key)
    if not v:continue
    def expand(v):
        if isinstance(v,list):return sum([expand(t) for t in v],[])
        q=pathlib.Path(v)
        if not q.is_absolute():q=pathlib.Path(cfg['path'])/q
        if q.suffix=='.txt':
            result['text_files'][str(q.relative_to(root))]=q.read_text() if q.is_relative_to(root) else ''
            return [pathlib.Path(x.strip()).name for x in q.read_text().splitlines() if x.strip()]
        return [x.name for x in q.glob('*.jpg')]
    values=expand(v); sets[name]=set(values)
    result['splits'][name]={'count':len(values),'unique':len(sets[name]),'list_hash':hashlib.sha256('\n'.join(sorted(values)).encode()).hexdigest(),'yaml_sha256':sha(p),'field':key}
result['intersections']={a+' / '+b:len(sets[a]&sets[b]) for a in sets for b in sets if a<b}
for p in sorted(weights):
    with contextlib.redirect_stdout(io.StringIO()):
        m=YOLO(p).model.float().cpu().eval()
        example=torch.zeros(1,3,640,640)
        tpmacs,_=tp.utils.count_ops_and_params(copy.deepcopy(m),example)
        thopmacs,_=profile(copy.deepcopy(m),inputs=(example,),verbose=False)
        shapes={k:list(v.shape) for k,v in m.state_dict().items()}
        result['checkpoints'][p]={'sha256':sha(p),'bytes':pathlib.Path(p).stat().st_size,'parameters':sum(t.numel() for t in m.parameters()),'tp_gmac':float(tpmacs)/1e9,'thop_gmac':float(thopmacs)/1e9,'dtype':str(next(m.parameters()).dtype),'shape_hash':hashlib.sha256(json.dumps(shapes,sort_keys=True).encode()).hexdigest(),'training_modules':[n for n,_ in m.named_modules() if 'distiller' in n or 'projector' in n or '_kd' in n]}
    print('checked '+p,file=sys.stderr,flush=True)
# raw metadata and latency output if retained
for pattern in ['runs/prune/experiment17_tiered_taylor_coco2017/20260915_123525/run_info.json','runs/recovery/experiment19_bn_recalibration_coco2017/20260915_133506/run_info.json','reports/*latency*','*latency*.log']:
    for p in root.glob(pattern):
        if p.is_file():result['text_files'][str(p.relative_to(root))]=p.read_text(errors='replace')
print(json.dumps(result,ensure_ascii=False))
