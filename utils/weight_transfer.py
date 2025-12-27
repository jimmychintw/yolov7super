"""
Weight Transfer Utility for 1B4H Architecture

Handles loading pre-trained weights from 1B1H models to 1B4H models.
Only loads Backbone and Neck layers, discards incompatible Head layers.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def load_transfer_weights(model, weights_path, device):
    """
    Load pre-trained weights with shape matching (transfer learning).

    Loads weights from a checkpoint file, keeping only layers where both
    name and shape match the target model. This effectively loads Backbone
    and Neck weights while discarding incompatible Head weights.

    Args:
        model: Target model (1B4H architecture)
        weights_path: Path to source checkpoint (.pt file)
        device: Target device for loading weights

    Returns:
        tuple: (transferred_count, total_count) - number of transferred items
               and total items in target model
    """
    # 1. Load checkpoint
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model' in ckpt:
        source_model = ckpt['model']
        # Handle both nn.Module and state_dict
        if hasattr(source_model, 'state_dict'):
            state_dict = source_model.float().state_dict()
        else:
            state_dict = source_model
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        # Assume checkpoint is the state_dict itself
        state_dict = ckpt

    # 2. Get current model structure
    model_state_dict = model.state_dict()

    # 3. Filter weights (Intersect)
    # Only keep weights where name and shape match exactly
    # This automatically filters out Detect Head (different channel count)
    intersect_dict = {}
    skipped_keys = []

    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                intersect_dict[k] = v
            else:
                skipped_keys.append((k, v.shape, model_state_dict[k].shape))
        else:
            # Key not in target model (could be renamed or removed)
            skipped_keys.append((k, v.shape, None))

    # 4. Load weights
    model.load_state_dict(intersect_dict, strict=False)

    # 5. Log transfer results
    transferred = len(intersect_dict)
    total = len(model_state_dict)

    logger.info(f'Transferred {transferred}/{total} items from {weights_path}')
    logger.info(f'Backbone + Neck loaded, Head initialized from scratch')

    # Log skipped layers for debugging
    if skipped_keys:
        head_skipped = [k for k, _, _ in skipped_keys if 'model.77' in k or 'heads' in k]
        other_skipped = [k for k, src, tgt in skipped_keys if 'model.77' not in k and 'heads' not in k]

        if head_skipped:
            logger.info(f'Skipped {len(head_skipped)} Head layers (expected - shape mismatch)')
        if other_skipped:
            logger.warning(f'Skipped {len(other_skipped)} other layers: {other_skipped[:5]}...')

    return transferred, total


def get_transfer_summary(weights_path, device='cpu'):
    """
    Analyze a checkpoint file and return summary of its contents.

    Useful for debugging and understanding what weights will be transferred.

    Args:
        weights_path: Path to checkpoint file
        device: Device for loading

    Returns:
        dict: Summary containing layer counts, shapes, etc.
    """
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    if 'model' in ckpt:
        source_model = ckpt['model']
        if hasattr(source_model, 'state_dict'):
            state_dict = source_model.float().state_dict()
        else:
            state_dict = source_model
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Categorize layers
    backbone_layers = [k for k in state_dict.keys() if 'model.' in k and int(k.split('.')[1]) < 50]
    neck_layers = [k for k in state_dict.keys() if 'model.' in k and 50 <= int(k.split('.')[1]) < 77]
    head_layers = [k for k in state_dict.keys() if 'model.77' in k or 'heads' in k]
    other_layers = [k for k in state_dict.keys() if k not in backbone_layers + neck_layers + head_layers]

    return {
        'total_layers': len(state_dict),
        'backbone_layers': len(backbone_layers),
        'neck_layers': len(neck_layers),
        'head_layers': len(head_layers),
        'other_layers': len(other_layers),
        'checkpoint_keys': list(ckpt.keys()) if isinstance(ckpt, dict) else ['state_dict'],
    }
