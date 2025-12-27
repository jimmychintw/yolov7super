#!/usr/bin/env python
"""
Test script for Auto LR Scaling feature
Reference: SDD v1.3, Section 2.8
"""
import yaml
import sys
sys.path.insert(0, '.')


def test_auto_lr_scaling(batch_size):
    """模擬 train.py 中的 Auto LR Scaling 邏輯"""
    # 載入 hyp
    with open("data/hyp.scratch.tiny.noota.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    nbs = 64  # nominal batch size
    current_bs = batch_size

    print(f"=== Test Batch Size: {batch_size} ===")
    print(f"Original LR0: {hyp['lr0']}")
    print(f"Original Warmup Epochs: {hyp.get('warmup_epochs', 3.0)}")

    if current_bs > nbs:
        scale_factor = (current_bs / nbs) ** 0.5
        original_lr = hyp['lr0']
        hyp['lr0'] *= scale_factor

        if current_bs >= 128:
            hyp['warmup_epochs'] = max(hyp.get('warmup_epochs', 3.0), 5.0)

        print(f"[Auto-LR] Batch Size {current_bs} > {nbs}. Scaling LR by {scale_factor:.4f}x")
        print(f"[Auto-LR] LR0: {original_lr:.4f} -> {hyp['lr0']:.4f}")
        print(f"[Auto-LR] Warmup Epochs: {hyp.get('warmup_epochs', 3.0)}")
    else:
        print("No scaling needed (batch <= 64)")
    print()

    return hyp['lr0'], hyp.get('warmup_epochs', 3.0)


if __name__ == '__main__':
    print("=" * 60)
    print("Auto LR Scaling Test (Square Root Strategy)")
    print("Reference: SDD v1.3, Section 2.8")
    print("=" * 60)
    print()

    # 測試不同 batch size
    expected = {
        64: (0.01, 3.0),      # No scaling
        128: (0.0141, 5.0),   # sqrt(128/64) = 1.414
        256: (0.02, 5.0),     # sqrt(256/64) = 2.0
        384: (0.0245, 5.0),   # sqrt(384/64) = 2.449
    }

    all_pass = True
    for bs in [64, 128, 256, 384]:
        lr, warmup = test_auto_lr_scaling(bs)
        exp_lr, exp_warmup = expected[bs]

        # 允許 0.001 的誤差
        lr_ok = abs(lr - exp_lr) < 0.001
        warmup_ok = warmup == exp_warmup

        if lr_ok and warmup_ok:
            print(f"✓ Batch {bs}: PASS")
        else:
            print(f"✗ Batch {bs}: FAIL (expected LR={exp_lr}, warmup={exp_warmup})")
            all_pass = False
        print()

    print("=" * 60)
    if all_pass:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
