import time, torch
from ultralytics import YOLO

def bench(weights, name, iters=200):
    model = YOLO(weights)
    m = model.model.eval().cuda()
    try:
        params = sum(p.numel() for p in m.parameters())
    except Exception:
        params = -1
    x = torch.zeros(1, 3, 640, 640, device="cuda")
    with torch.no_grad():
        for _ in range(30):
            m(x)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            m(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    times.sort()
    mean = sum(times) / len(times)
    median = times[len(times)//2]
    p90 = times[int(len(times)*0.9)]
    print(f"{name}: params={params} mean={mean:.2f}ms median={median:.2f}ms p90={p90:.2f}ms min={times[0]:.2f}ms max={times[-1]:.2f}ms", flush=True)
    del model, m, x
    torch.cuda.empty_cache()

if __name__ == "__main__":
    bench("/root/YOLO/weights/yolo11s.pt", "yolo11s_official")
    bench("/root/YOLO/runs/recovery/experiment19b_coco2017/node15_yolo11s_kd/20260917_101816/training/weights/best_clean.pt", "node15_15pct")
    bench("/root/YOLO/runs/recovery/experiment19b_coco2017/node20_yolo11s_kd/20260917_105730/training/weights/best_clean.pt", "node20_20pct")
