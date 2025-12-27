"""
Phase 3: 訓練迴圈細粒度效能剖析
使用 CUDA Events 精確測量各階段時間
"""
import sys
import time
from pathlib import Path
from argparse import Namespace
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.cuda.amp as amp
import yaml
import numpy as np

from models.yolo import Model
from utils.datasets import create_dataloader
from utils.loss import ComputeLoss, ComputeLossOTA


class CUDAProfiler:
    """使用 CUDA Events 精確測量 GPU 時間"""
    def __init__(self):
        self.timings = {}
        self.start_events = {}
        self.end_events = {}

    def start(self, name):
        if name not in self.start_events:
            self.start_events[name] = torch.cuda.Event(enable_timing=True)
            self.end_events[name] = torch.cuda.Event(enable_timing=True)
            self.timings[name] = []
        self.start_events[name].record()

    def end(self, name):
        self.end_events[name].record()

    def sync_and_record(self):
        torch.cuda.synchronize()
        for name in self.start_events:
            if name in self.end_events:
                elapsed = self.start_events[name].elapsed_time(self.end_events[name])
                self.timings[name].append(elapsed)

    def report(self, skip_first=5):
        print("\n" + "="*60)
        print("Profile Result - CUDA Event Timing (ms)")
        print("="*60)
        total = 0
        for name, times in self.timings.items():
            if len(times) > skip_first:
                times = times[skip_first:]
                avg = np.mean(times)
                std = np.std(times)
                total += avg
                print(f"{name:25s}: {avg:8.2f} ± {std:6.2f} ms")
        print("-"*60)
        print(f"{'Total':25s}: {total:8.2f} ms")

        # GPU 利用率估算
        gpu_time = 0
        if 'forward' in self.timings:
            gpu_time += np.mean(self.timings['forward'][skip_first:])
        if 'backward' in self.timings:
            gpu_time += np.mean(self.timings['backward'][skip_first:])
        if total > 0:
            print(f"{'GPU Utilization Est':25s}: {gpu_time/total*100:6.1f}%")
        print("="*60)
        return self.timings


def profile_training(data_yaml, hyp_yaml, cfg, batch_size=384, workers=16,
                     n_iter=50, cache_images=True):
    """完整訓練迴圈 profiling"""

    device = torch.device('cuda:0')

    # 載入設定
    with open(hyp_yaml) as f:
        hyp = yaml.safe_load(f)

    with open(data_yaml) as f:
        data_dict = yaml.safe_load(f)
    train_path = data_dict['train']
    nc = int(data_dict['nc'])

    print(f"\n{'='*60}")
    print(f"Training Loop Profiler")
    print(f"{'='*60}")
    print(f"Model: {cfg}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {workers}")
    print(f"Cache images: {cache_images}")
    print(f"Mosaic: {hyp.get('mosaic', 1.0)}")
    print(f"Iterations: {n_iter}")
    print(f"{'='*60}\n")

    # 建立模型
    print("Loading model...")
    model = Model(cfg, ch=3, nc=nc).to(device)
    model.nc = nc
    model.hyp = hyp  # ComputeLossOTA 需要此屬性
    model.gr = 1.0   # iou loss ratio
    model.train()

    # 建立 opt 物件 (模擬 train.py 的 opt)
    opt = Namespace(single_cls=False)

    # 建立 DataLoader
    print("Creating dataloader...")
    dataloader, dataset = create_dataloader(
        train_path, 320, batch_size, 32, opt, hyp=hyp,
        augment=True, cache=cache_images, rect=False,
        workers=workers, image_weights=False, quad=False,
        prefix='profile: '
    )

    # 建立 Loss 和 Optimizer
    use_ota = hyp.get('loss_ota', 1) == 1
    print(f"Using {'ComputeLossOTA' if use_ota else 'ComputeLoss'}")
    if use_ota:
        compute_loss = ComputeLossOTA(model)
    else:
        compute_loss = ComputeLoss(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = amp.GradScaler()

    # Warmup
    print("Warming up...")
    data_iter = iter(dataloader)
    for _ in range(3):
        imgs, targets, _, _ = next(data_iter)
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)
        with amp.autocast():
            pred = model(imgs)
            if use_ota:
                loss, _ = compute_loss(pred, targets, imgs)
            else:
                loss, _ = compute_loss(pred, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Profiling
    print(f"\nProfiling {n_iter} iterations...")
    profiler = CUDAProfiler()
    data_wait_times = []

    data_iter = iter(dataloader)
    for i in range(n_iter):
        # 1. DataLoader (CPU 等待時間)
        t0 = time.perf_counter()
        imgs, targets, paths, _ = next(data_iter)
        t1 = time.perf_counter()
        data_wait_times.append((t1 - t0) * 1000)

        # 2. Data Transfer
        profiler.start('transfer')
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)
        profiler.end('transfer')

        # 3. Forward
        profiler.start('forward')
        with amp.autocast():
            pred = model(imgs)
        profiler.end('forward')

        # 4. Loss
        profiler.start('loss')
        with amp.autocast():
            if use_ota:
                loss, loss_items = compute_loss(pred, targets, imgs)
            else:
                loss, loss_items = compute_loss(pred, targets)
        profiler.end('loss')

        # 5. Backward
        profiler.start('backward')
        scaler.scale(loss).backward()
        profiler.end('backward')

        # 6. Optimizer
        profiler.start('optimizer')
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        profiler.end('optimizer')

        # 記錄
        profiler.sync_and_record()

        if (i + 1) % 10 == 0:
            print(f"  Iteration {i+1}/{n_iter}")

    # 報告
    print("\n" + "="*60)
    print("DataLoader Wait Time (CPU preparation)")
    print("="*60)
    data_wait = np.array(data_wait_times[5:])
    print(f"{'dataloader_wait':25s}: {data_wait.mean():8.2f} ± {data_wait.std():6.2f} ms")

    profiler.report()

    return profiler.timings, data_wait_times


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco320.yaml')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.tiny.yaml')
    parser.add_argument('--cfg', type=str, default='cfg/training/yolov7-tiny-320.yaml')
    parser.add_argument('--batch-size', type=int, default=384)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--n-iter', type=int, default=50)
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()

    profile_training(
        args.data, args.hyp, args.cfg,
        batch_size=args.batch_size,
        workers=args.workers,
        n_iter=args.n_iter,
        cache_images=not args.no_cache
    )
