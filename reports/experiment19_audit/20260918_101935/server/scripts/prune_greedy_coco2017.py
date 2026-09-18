"""COCO2017: cost-aware greedy structural pruning and recovery.

Each trial copies the current accepted model; only the best feasible action is
kept. Prior experiment04 artifacts are read-only. Every invocation has its own
timestamped directory and records failures as well as successful candidates.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, C2PSA
from ultralytics.utils import LOGGER

from prune_independent_compare import choose_indices, read_candidates

ROOT = Path(__file__).resolve().parents[1]
TRIAL_FIELDS = ["step", "layer", "status", "reason", "before_channels", "after_channels",
                "indices", "parameters", "gmacs", "map50", "map50_95", "step_map_drop",
                "step_compute_saved", "total_compute_saved", "score", "elapsed_seconds"]


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def append_csv(path, fields, row):
    fresh = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fresh:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


class Tee:
    def __init__(self, stream, logfile):
        self.stream, self.logfile = stream, logfile
    def write(self, text):
        self.stream.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)
    def flush(self):
        self.stream.flush()
        self.logfile.flush()
    def isatty(self):
        return False
    @property
    def encoding(self):
        return "utf-8"


def shape_signature(model):
    return {k: list(v.shape) for k, v in model.state_dict().items()}


def fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stats(model, example):
    # TP's operation counter changes training flags on its copy, not the source.
    macs, _ = tp.utils.count_ops_and_params(copy.deepcopy(model).eval(), example)
    return sum(p.numel() for p in model.parameters()), float(macs) / 1e9


def save_model(model, path, base_weights):
    snapshot = copy.deepcopy(model).cpu().half()
    snapshot.zero_grad(set_to_none=True)
    if hasattr(snapshot, "criterion"):
        snapshot.criterion = None
    # Preserve topology AND tensors. Do not transplant tensors into original YAML.
    checkpoint = {
        "model": snapshot, "ema": None, "epoch": -1, "optimizer": None,
        "best_fitness": None, "updates": None, "date": now(),
        "version": importlib.metadata.version("ultralytics"),
        "train_args": {**dict(model.args), "model": str(path), "task": "detect"},
        "pruning_source": str(base_weights),
    }
    torch.save(checkpoint, path)


def safe_prune(model, name, example, initial_channels, ratio=0.125):
    model.eval().float()
    for p in model.parameters():
        p.requires_grad_(True)
    modules = dict(model.named_modules())
    reverse = {module: n for n, module in modules.items()}
    root = modules[name]
    indices = choose_indices(root, ratio, 8)
    # CSP split widths are coupled, and vanilla TP may silently mishandle chunk.
    split_outputs = {m.cv1.conv for m in model.modules() if isinstance(m, (C2f, C2PSA))}
    before = shape_signature(model)
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=example)
    group = graph.get_pruning_group(root, tp.prune_conv_out_channels, idxs=indices)
    if not graph.check_pruning_group(group):
        raise ValueError("dependency_group_rejected")
    operations = []
    for dep, idxs in group:
        module = dep.target.module
        module_name = reverse.get(module, "")
        out_prune = graph.is_out_channel_pruning_fn(dep.handler)
        if module in split_outputs and out_prune:
            raise ValueError(f"protect_CSP_chunk_width: {module_name}")
        if module_name == "model.0" or module_name.startswith(("model.0.", "model.10.")):
            raise ValueError(f"protect_stem_or_attention: {module_name}")
        if module_name.startswith("model.23.dfl"):
            raise ValueError(f"protect_DFL: {module_name}")
        if isinstance(module, torch.nn.Conv2d):
            # Final detection channels retain class/box semantics. Internal head
            # inputs (and coupled depthwise widths) may follow pruned neck outputs.
            if module_name.startswith("model.23.") and module_name.count(".") == 4 and out_prune:
                raise ValueError(f"protect_detection_output: {module_name}")
            if out_prune and module.out_channels - len(set(idxs)) < max(8, initial_channels[module_name] // 2):
                raise ValueError(f"minimum_remaining_channels: {module_name}")
            operations.append({"layer": module_name, "direction": "out" if out_prune else "in", "indices": list(idxs)})
    width = root.out_channels
    group.prune()
    after = shape_signature(model)
    changes = {k: {"before": before[k], "after": after[k]} for k in before if before[k] != after[k]}
    if not changes:
        raise ValueError("no_tensor_shape_change")
    for side in (320, 640):
        with torch.no_grad():
            prediction = model(torch.zeros(1, 3, side, side, device=example.device))[0]
        if prediction.shape[1] != 84 or not torch.isfinite(prediction).all():
            raise ValueError("invalid_detection_output")
    model.zero_grad(set_to_none=True)
    return {"before_channels": width, "after_channels": root.out_channels,
            "indices": indices, "dependencies": operations, "changed_tensors": changes}


def backward_check(model, device):
    trial = copy.deepcopy(model).train().float().to(device)
    for module in trial.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    for p in trial.parameters():
        p.requires_grad_(True)
    prediction = trial(torch.zeros(2, 3, 320, 320, device=device))
    def tensors(obj):
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, dict):
            for value in obj.values():
                yield from tensors(value)
        elif isinstance(obj, (tuple, list)):
            for value in obj:
                yield from tensors(value)
    loss = sum(t.float().square().mean() for t in tensors(prediction))
    if not torch.isfinite(loss):
        raise ValueError("nonfinite_forward")
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in trial.parameters() if p.grad is not None):
        raise ValueError("nonfinite_backward")


def report(run_dir, info, comparisons):
    lines = ["# YOLO11s COCO2017 structural pruning", "",
             "Search uses COCO val2017 for model selection; results are development-set metrics.",
             "Sensitivity masking is a heuristic, not exact channel-removal equivalence.",
             "GMAC reduction is measured on unfused models with the same counter.",
             f"Stop: {info['stop_reason']}",
             f"Compute reduction: {info['raw']['compute_reduction']:.2%}",
             f"Raw mAP50-95: {info['raw']['map50_95']:.6f}",
             "", "See run_info.json, steps.json and comparison.csv for full results."]
    (run_dir / 'report.md').write_text(chr(10).join(lines), encoding='utf-8')


def run(args, run_dir, info):
    from greedy_evaluation_coco2017 import evaluate, run_finetune, benchmark
    base = Path(args.weights).resolve()
    data = Path(args.data).resolve()
    sens = Path(args.sensitivity).resolve()
    for path in (base, data, sens):
        if not path.is_file():
            raise FileNotFoundError(path)
    target = args.target_reduction
    info.update({"target_compute_reduction": target, "configuration": vars(args),
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in (base, data, sens, Path(__file__), ROOT/'scripts/greedy_evaluation_coco2017.py', ROOT/'scripts/prune_independent_compare.py')},
        "environment": {k: importlib.metadata.version(k) for k in ('torch', 'ultralytics', 'torch-pruning')},
        "python": sys.version, "platform": platform.platform(), "gpu": torch.cuda.get_device_name(int(args.device)), "steps": []})
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device(f"cuda:{args.device}")
    model = YOLO(str(base)).model.float().to(device).eval()
    initial_channels = {n: m.out_channels for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)}
    candidates = [r['layer'] for r in read_candidates(sens)][:args.top_k]
    if not candidates:
        raise RuntimeError('No eligible sensitivity candidates')
    info['candidates'] = candidates
    example = torch.zeros(1, 3, 640, 640, device=device)
    baseline_params, baseline_gmacs = stats(model, example)
    original_eval = evaluate(base, data, run_dir / 'validation/baseline', args.device)
    info['baseline'] = {**original_eval, 'parameters': baseline_params, 'gmacs': baseline_gmacs}
    current_map, current_gmacs = original_eval['map50_95'], baseline_gmacs
    current_raw_eval = original_eval
    info['status'] = 'searching'
    write_json(run_dir / 'run_info.json', info)
    print(f"Baseline mAP={current_map:.6f}, GMACs={baseline_gmacs:.6f}; {len(candidates)} candidates; target={target:.4%}", flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    search_started = time.perf_counter()
    stop_reason = 'max_steps'
    for step in range(1, args.max_steps + 1):
        best_model, best_row = None, None
        parent_hash = fingerprint(model)
        step_rows = []
        for name in candidates:
            started = time.perf_counter()
            trial = copy.deepcopy(model)
            row = {'step': step, 'layer': name, 'status': 'ok', 'reason': ''}
            try:
                change = safe_prune(trial, name, example, initial_channels)
                params, gmacs = stats(trial, example)
                gain = (current_gmacs - gmacs) / baseline_gmacs
                if gain <= 1e-6:
                    raise ValueError('no_compute_reduction')
                save_model(trial, run_dir / '_candidate.pt', base)
                measured = evaluate(run_dir / '_candidate.pt', data, run_dir / 'validation/candidate', args.device)
                if not all(math.isfinite(float(x)) for x in measured.values()):
                    raise ValueError('nonfinite_metric')
                if original_eval['map50_95'] - measured['map50_95'] > args.max_map_drop:
                    raise ValueError('cumulative_accuracy_drop_limit')
                drop = current_map - measured['map50_95']
                row.update(change)
                row.update({'parameters': params, 'gmacs': gmacs, **measured, 'step_map_drop': drop,
                    'step_compute_saved': gain, 'total_compute_saved': 1 - gmacs / baseline_gmacs,
                    'score': drop / gain})
                if best_row is None or (row['score'], -gain, name) < (best_row['score'], -best_row['step_compute_saved'], best_row['layer']):
                    backward_check(trial, device)
                    best_model, best_row = copy.deepcopy(trial), dict(row)
            except (RuntimeError, ValueError, IndexError, KeyError, AssertionError) as error:
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    raise
                row.update(status='rejected', reason=f'{type(error).__name__}: {error}')
                print(f"REJECT step={step} {name}: {row['reason']}", flush=True)
            row['elapsed_seconds'] = time.perf_counter() - started
            append_csv(run_dir / 'candidates.csv', TRIAL_FIELDS, row)
            step_rows.append(row)
            del trial
            gc.collect()
        assert fingerprint(model) == parent_hash, 'trial mutated parent checkpoint'
        if best_model is None:
            stop_reason = 'no_feasible_candidates'
            break
        best_row['parent_state_sha256'] = parent_hash
        model = best_model
        current_map, current_gmacs = best_row['map50_95'], best_row['gmacs']
        info['steps'].append(best_row)
        save_model(model, run_dir / 'pruned_raw.pt', base)
        write_json(run_dir / 'steps.json', info['steps'])
        write_json(run_dir / 'run_info.json', info)
        print(f"ACCEPT step={step} {best_row['layer']} {best_row['before_channels']}->{best_row['after_channels']}; mAP={current_map:.6f}; GMAC saved={best_row['total_compute_saved']:.4%}", flush=True)
        if best_row['total_compute_saved'] >= target:
            stop_reason = 'target_reached'
            break
    if not info['steps']:
        raise RuntimeError('No safe pruning step found; see candidates.csv')
    info['stop_reason'] = stop_reason
    info['search_seconds'] = time.perf_counter() - search_started
    info['search_peak_allocated_mb'] = torch.cuda.max_memory_allocated(device) / 1e6
    reloaded = YOLO(str(run_dir / 'pruned_raw.pt')).model
    assert shape_signature(reloaded) == shape_signature(model), 'saved structure mismatch'
    raw_eval = evaluate(run_dir / 'pruned_raw.pt', data, run_dir/'validation/greedy_raw', args.device)
    assert abs(raw_eval['map50_95'] - current_map) < 1e-5, 'selected/reloaded mAP mismatch'
    params, gmacs = stats(model, example)
    info['raw'] = {**raw_eval, 'parameters': params, 'gmacs': gmacs, 'compute_reduction': 1-gmacs/baseline_gmacs, 'parameter_reduction': 1-params/baseline_params}
    (run_dir / '_candidate.pt').unlink(missing_ok=True)
    del model, best_model, reloaded
    gc.collect()
    torch.cuda.empty_cache()
    info['status'] = 'finetuning'
    write_json(run_dir / 'run_info.json', info)
    if args.epochs == 0:
        info.update(status='complete', finished_at=now(), elapsed_seconds=time.perf_counter()-info.pop('_timer'))
        write_json(run_dir/'run_info.json', info)
        report(run_dir, info, [])
        print('COMPLETE ' + str(run_dir), flush=True)
        return
    weights = {'greedy': run_dir/'pruned_raw.pt'}
    raw_maps = {'greedy': raw_eval}
    comparisons, best_weights = [], {'baseline': base}
    for label, source in weights.items():
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        best = run_finetune(source, data, run_dir/'finetune'/label, args.device, epochs=args.epochs)
        measured = evaluate(best, data, run_dir/'validation'/label, args.device)
        row = {'label': label, 'raw_map50_95': raw_maps[label]['map50_95'], **measured,
               'finetune_seconds': time.perf_counter()-started, 'peak_allocated_mb': torch.cuda.max_memory_allocated(device)/1e6,
               'best_weights': str(best)}
        best_weights[label] = best
        comparisons.append(row)
        info['finetuning'] = comparisons
        write_json(run_dir/'run_info.json', info)
        print(f"FINETUNED {label}: mAP={measured['map50_95']:.6f}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    performance = benchmark(best_weights, args.device)
    info['benchmark'] = performance
    assert sha(base) == info['sources'][str(base.relative_to(ROOT))], 'baseline file changed'
    for row in comparisons:
        row.update(performance[row['label']])
        row['parameter_reduction'] = 1-row['parameters']/baseline_params
        row['compute_reduction'] = 1-row['gmacs']/baseline_gmacs
        row['model_mb'] = Path(row['best_weights']).stat().st_size/1e6
        append_csv(run_dir/'comparison.csv', list(row), row)
    info.update(status='complete', finished_at=now(), elapsed_seconds=time.perf_counter()-info.pop('_timer'), comparisons=comparisons)
    write_json(run_dir/'run_info.json', info)
    report(run_dir, info, comparisons)
    print('COMPLETE ' + str(run_dir), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='0')
    parser.add_argument('--max-steps', default=20, type=int)
    parser.add_argument('--weights', default=str(ROOT/'weights/baseline/yolo11s_coco2017_best.pt'))
    parser.add_argument('--data', default=str(ROOT/'configs/coco2017.yaml'))
    parser.add_argument('--sensitivity', default=str(ROOT/'reports/experiment09_yolo11s_coco2017_sensitivity.csv'))
    parser.add_argument('--target-reduction', type=float, default=0.20)
    parser.add_argument('--max-map-drop', type=float, default=0.03)
    parser.add_argument('--top-k', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--run-label', default='experiment10_coco2017_greedy')
    args = parser.parse_args()
    if not 0 < args.target_reduction < 1 or args.top_k < 1 or args.epochs < 0 or args.max_steps < 1:
        parser.error('Invalid search settings')
    os.chdir(ROOT)
    run_dir = ROOT / 'runs/prune' / args.run_label / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=False)
    info = {'started_at': now(), 'status': 'starting', '_timer': time.perf_counter()}
    original_out, original_err = sys.stdout, sys.stderr
    with (run_dir / 'run.log').open('w', encoding='utf-8', buffering=1) as logfile:
        sys.stdout, sys.stderr = Tee(original_out, logfile), Tee(original_err, logfile)
        handlers = [(h, h.stream) for h in LOGGER.handlers if hasattr(h, 'stream')]
        for h, _ in handlers:
            h.setStream(sys.stdout)
        try:
            print('RUN_DIR ' + str(run_dir), flush=True)
            run(args, run_dir, info)
        except BaseException as error:
            info.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error), finished_at=now())
            if '_timer' in info:
                info['elapsed_seconds'] = time.perf_counter()-info.pop('_timer')
            write_json(run_dir/'run_info.json', info)
            traceback.print_exc()
            raise
        finally:
            for h, stream in handlers:
                h.setStream(stream)
            sys.stdout, sys.stderr = original_out, original_err


if __name__ == '__main__':
    main()
