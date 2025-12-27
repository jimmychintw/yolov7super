#!/usr/bin/env python3
"""
RL-based Data Augmentation Search using Optuna
Phase 4: Automatic hyperparameter optimization for each head

Usage:
    python utils/rl_augment.py --head 0 --trials 30  # Optimize Head 0
    python utils/rl_augment.py --head 1 --trials 30  # Optimize Head 1
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

try:
    import optuna
except ImportError:
    print("Please install optuna: pip install optuna")
    sys.exit(1)


def load_head_config(config_path: str) -> dict:
    """Load head configuration and return class assignments per head."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    head_classes = {}
    for head_key, head_data in config.get('head_assignments', {}).items():
        head_idx = int(head_key.split('_')[1])
        head_classes[head_idx] = {
            'name': head_data.get('name', f'Head {head_idx}'),
            'classes': head_data.get('classes', [])
        }
    return head_classes


def create_trial_hyp(trial: optuna.Trial, base_hyp_path: str, output_path: str) -> dict:
    """Create hyperparameter file for this trial with sampled augmentation values."""
    # Load base hyperparameters
    with open(base_hyp_path, 'r') as f:
        hyp = yaml.safe_load(f)

    # Sample augmentation hyperparameters
    hyp['degrees'] = trial.suggest_float('degrees', 0.0, 45.0)
    hyp['flipud'] = trial.suggest_float('flipud', 0.0, 0.5)
    hyp['fliplr'] = trial.suggest_float('fliplr', 0.0, 0.5)
    hyp['shear'] = trial.suggest_float('shear', 0.0, 5.0)
    hyp['mixup'] = trial.suggest_float('mixup', 0.0, 0.2)

    # Save trial hyperparameters
    with open(output_path, 'w') as f:
        yaml.dump(hyp, f, default_flow_style=False)

    return hyp


def run_proxy_training(
    weights: str,
    data: str,
    cfg: str,
    hyp: str,
    head_config: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    freeze: int,
    device: str,
    project: str,
    name: str,
    per_class_ap_path: str
) -> bool:
    """Run proxy training and return success status."""
    cmd = [
        sys.executable, 'train.py',
        '--img-size', str(img_size), str(img_size),
        '--batch-size', str(batch_size),
        '--epochs', str(epochs),
        '--weights', weights,
        '--data', data,
        '--cfg', cfg,
        '--hyp', hyp,
        '--device', device,
        '--project', project,
        '--name', name,
        '--freeze', str(freeze),
        '--transfer-weights',
        '--noautoanchor',
        '--cache-images',
        '--heads', '4',
        '--head-config', head_config,
        '--exist-ok',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("Training timeout")
        return False
    except Exception as e:
        print(f"Training error: {e}")
        return False

    # Run validation to get per-class AP
    val_cmd = [
        sys.executable, 'test.py',
        '--weights', f'{project}/{name}/weights/last.pt',
        '--data', data,
        '--batch-size', str(batch_size),
        '--img-size', str(img_size),
        '--device', device,
        '--save-per-class-ap', per_class_ap_path,
        '--task', 'val'
    ]

    try:
        result = subprocess.run(val_cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def calculate_head_map(per_class_ap_path: str, head_classes: list) -> float:
    """Calculate mAP for specific head classes."""
    if not os.path.exists(per_class_ap_path):
        return 0.0

    with open(per_class_ap_path, 'r') as f:
        data = json.load(f)

    per_class = data.get('per_class', {})

    # Calculate mean AP@0.5 for head classes
    ap_values = []
    for cls_id in head_classes:
        cls_key = str(cls_id)
        if cls_key in per_class:
            ap_values.append(per_class[cls_key]['ap50'])

    if not ap_values:
        return 0.0

    return sum(ap_values) / len(ap_values)


def create_objective(args, head_classes: dict):
    """Create Optuna objective function for the specified head."""
    target_head = args.head
    target_classes = head_classes[target_head]['classes']
    head_name = head_classes[target_head]['name']
    total_trials = args.trials

    # Progress tracking
    progress = {
        'start_time': time.time(),
        'trial_times': [],
        'best_score': 0.0
    }

    print(f"\n{'='*60}")
    print(f"Optimizing Head {target_head}: {head_name}")
    print(f"Classes: {target_classes}")
    print(f"Total trials: {total_trials}")
    print(f"{'='*60}\n")

    def objective(trial: optuna.Trial) -> float:
        trial_start = time.time()
        trial_name = f"rl_head{target_head}_trial{trial.number}"

        # Create trial hyperparameters
        trial_hyp_path = f"temp/hyp.{trial_name}.yaml"
        os.makedirs('temp', exist_ok=True)

        hyp = create_trial_hyp(trial, args.base_hyp, trial_hyp_path)

        # Progress info
        elapsed = time.time() - progress['start_time']
        elapsed_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m"
        if progress['trial_times']:
            avg_time = sum(progress['trial_times']) / len(progress['trial_times'])
            remaining = avg_time * (total_trials - trial.number)
            eta_str = f"{int(remaining//3600)}h {int((remaining%3600)//60)}m"
        else:
            eta_str = "calculating..."

        print(f"\n{'='*60}")
        print(f"[Trial {trial.number + 1}/{total_trials}] Head {target_head} | Elapsed: {elapsed_str} | ETA: {eta_str}")
        print(f"Best so far: {progress['best_score']:.4f}")
        print(f"{'='*60}")
        print(f"degrees={hyp['degrees']:.2f}, flipud={hyp['flipud']:.3f}, "
              f"fliplr={hyp['fliplr']:.3f}, shear={hyp['shear']:.2f}, mixup={hyp['mixup']:.3f}")

        # Run proxy training
        per_class_ap_path = f"temp/{trial_name}_class_ap.json"

        success = run_proxy_training(
            weights=args.weights,
            data=args.data,
            cfg=args.cfg,
            hyp=trial_hyp_path,
            head_config=args.head_config,
            epochs=args.proxy_epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            freeze=args.freeze,
            device=args.device,
            project='runs/rl_search',
            name=trial_name,
            per_class_ap_path=per_class_ap_path
        )

        if not success:
            print(f"Trial {trial.number + 1} FAILED")
            progress['trial_times'].append(time.time() - trial_start)
            return 0.0

        # Calculate head-specific mAP
        head_map = calculate_head_map(per_class_ap_path, target_classes)

        # Update progress tracking
        trial_time = time.time() - trial_start
        progress['trial_times'].append(trial_time)
        if head_map > progress['best_score']:
            progress['best_score'] = head_map

        print(f"\n>>> Trial {trial.number + 1}/{total_trials} COMPLETE | mAP@0.5: {head_map:.4f} | Time: {trial_time/60:.1f}min")
        if head_map >= progress['best_score']:
            print(f">>> NEW BEST! <<<")

        # Cleanup trial files and directory
        if os.path.exists(trial_hyp_path):
            os.remove(trial_hyp_path)
        if os.path.exists(per_class_ap_path):
            os.remove(per_class_ap_path)
        trial_dir = f'runs/rl_search/{trial_name}'
        if os.path.exists(trial_dir):
            shutil.rmtree(trial_dir)

        return head_map

    return objective


def save_best_hyp(study: optuna.Study, args, head_idx: int):
    """Save best hyperparameters to file."""
    best_params = study.best_params

    # Load base hyp and update with best params
    with open(args.base_hyp, 'r') as f:
        hyp = yaml.safe_load(f)

    hyp.update(best_params)

    output_path = f"data/hyp.head{head_idx}.best.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(hyp, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"Best hyperparameters for Head {head_idx} saved to: {output_path}")
    print(f"Best mAP@0.5: {study.best_value:.4f}")
    print(f"Best params: {best_params}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='RL-based Data Augmentation Search')
    parser.add_argument('--head', type=int, required=True, help='Head index to optimize (0-3)')
    parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials')
    parser.add_argument('--head-config', type=str, default='data/coco_320_1b4h_geometry.yaml',
                        help='Head configuration file')
    parser.add_argument('--base-hyp', type=str, default='data/hyp.scratch.tiny.yaml',
                        help='Base hyperparameter file')
    parser.add_argument('--weights', type=str, default='backbone_elite_0.435.pt',
                        help='Pretrained backbone weights')
    parser.add_argument('--data', type=str, default='data/coco320.yaml',
                        help='Dataset configuration')
    parser.add_argument('--cfg', type=str, default='cfg/training/yolov7-tiny-1b4h.yaml',
                        help='Model configuration')
    parser.add_argument('--proxy-epochs', type=int, default=10,
                        help='Number of epochs for proxy training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for proxy training')
    parser.add_argument('--img-size', type=int, default=320,
                        help='Image size')
    parser.add_argument('--freeze', type=int, default=50,
                        help='Number of layers to freeze')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device')

    args = parser.parse_args()

    # Load head configuration
    head_classes = load_head_config(args.head_config)

    if args.head not in head_classes:
        print(f"Error: Head {args.head} not found. Available heads: {list(head_classes.keys())}")
        sys.exit(1)

    # Create Optuna study
    study_name = f"rl_augment_head{args.head}"
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        storage=f'sqlite:///temp/{study_name}.db',
        load_if_exists=True
    )

    # Run optimization
    objective = create_objective(args, head_classes)
    study.optimize(objective, n_trials=args.trials)

    # Save best hyperparameters
    save_best_hyp(study, args, args.head)


if __name__ == '__main__':
    main()
