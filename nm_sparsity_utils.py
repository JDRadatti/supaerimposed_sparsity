import os
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import vgg
import sparse


# ---------------------------------------------------------------------------
# SPARSITY MASK
# ---------------------------------------------------------------------------

def inject_switchable_mask(
    model: nn.Module,
    sparsity_key: str,
    N: int,
    M: int,
    conv_axis: str = "out",
    linear_orientation: str = "row",
):
    """
    Registers an N:M sparsity mask for a given key into eligible Conv2d / Linear layers.
    Skips the very first and very last such layers.

    Args:
        model:               The target ``nn.Module``.
        sparsity_key:        A unique identifier for this sparsity mode (e.g., "1:4").
        N:                   Number of weights to keep per group.
        M:                   Group size.
        conv_axis:           Axis for Conv2d mask (``"out"`` or ``"in"``).
        linear_orientation:  Orientation for Linear mask (``"row"`` or ``"col"``).
    """
    # Identify first & last Conv/Linear to skip
    conv_linear = [
        m
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
        and hasattr(m, "weight")
    ]
    first_last = set(conv_linear[:1] + conv_linear[-1:]) if conv_linear else set()

    for module in conv_linear:
        if module in first_last:
            continue

        # Create a dictionary to hold masks if it doesn't exist
        if not hasattr(module, 'nm_masks'):
            module.nm_masks = {}
            module.nm_masks['dense'] = None

        # Generate the N:M Mask
        if sparsity_key not in module.nm_masks:
            w = module.weight.detach()
            if isinstance(module, nn.Conv2d):
                mask = sparse._nm_mask_conv2d_axis(w, N, M, axis=conv_axis)
            else: # nn.Linear
                mask = sparse._nm_mask_linear(w, N, M, orientation=linear_orientation)
            module.nm_masks[sparsity_key] = mask.type_as(w)

        _patch_forward_for_masks(module)


def _patch_forward_for_masks(module: nn.Module) -> None:
    """Patch forward on a module that already has ``nm_masks`` set up.

    If the module already has ``nm_masks`` and isn't already patched, this
    overrides ``module.forward`` with a closure that applies the active mask.
    """
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return
    if not hasattr(module, 'nm_masks'):
        return

    if not hasattr(module, '_original_forward'):
        module._original_forward = module.forward

    if not hasattr(module, 'active_mask_key'):
        module.active_mask_key = 'dense'

    def _make_forward(m):
        def _forward(input):
            current_mask_key = getattr(m, 'active_mask_key', 'dense')

            if current_mask_key == 'dense':
                return m._original_forward(input)

            mask = m.nm_masks.get(current_mask_key)
            if mask is None:
                print(f"Warning: Mask '{current_mask_key}' not found for module. Falling back to dense.")
                return m._original_forward(input)

            masked_weight = m.weight * mask.to(m.weight.device)

            if isinstance(m, nn.Linear):
                return F.linear(input, masked_weight, m.bias)
            return F.conv2d(input, masked_weight, m.bias,
                            m.stride, m.padding, m.dilation, m.groups)
        return _forward

    module.__dict__['forward'] = _make_forward(module)


def _apply_active_mask(model: torch.nn.Module) -> None:
    """
    Multiply the *active* mask onto the underlying weight **in‑place**.
    Call this **after** optimizer.step().
    """
    for module in model.modules():
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            continue
        if not hasattr(module, "nm_masks"):
            continue

        active_key = getattr(module, "active_mask_key", "dense")
        if active_key == "dense":
            continue

        mask = module.nm_masks.get(active_key)
        if mask is None:
            continue

        module.weight.data.mul_(mask.to(module.weight.device))

# ---------------------------------------------------------------------------
# SPARSITY SETTERS/GETTERS
# ---------------------------------------------------------------------------

def set_active_sparsity(model: nn.Module, sparsity_key: str):
    """
    Sets the active sparsity mask key for all eligible modules in the model.
    'sparsity_key' can be a registered key (e.g., "1:4") or "dense".
    If 'dense' is passed, it ensures no specific mask is active, effectively
    restoring the module to its original forward behavior.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'nm_masks'):
            
            if sparsity_key == 'dense':
                if not hasattr(module, 'active_mask_key') or module.active_mask_key != 'dense':
                    module.active_mask_key = 'dense'
            elif sparsity_key in module.nm_masks:
                if not hasattr(module, 'active_mask_key') or module.active_mask_key != sparsity_key:
                    module.active_mask_key = sparsity_key
            else:
                pass # Keep the current active mask if the requested key isn't available.

def get_active_sparsity(model: nn.Module) -> str:
    """
    Returns the active sparsity key for the first eligible module found that has 'active_mask_key'.
    Assumes all eligible modules in a model are configured to use the same active mask.
    Returns 'dense' if no specific mask is active or if no configured modules are found.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'active_mask_key'):
            return module.active_mask_key
    return "dense"


# ---------------------------------------------------------------------------
# BN BUFFERS (per-sparsity-key)
# ---------------------------------------------------------------------------

_bn_buffers: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
"""Maps sparsity_key -> module_name -> {running_mean, running_var, num_batches_tracked}."""


def _collect_bn_stats(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    """Capture current BN running stats for every BatchNorm module.

    Returns:
        ``{module_name: {"running_mean": tensor, "running_var": tensor,
                         "num_batches_tracked": tensor}}``
    """
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            stats[name] = {
                "running_mean": module.running_mean.detach().clone(),
                "running_var": module.running_var.detach().clone(),
                "num_batches_tracked": module.num_batches_tracked.detach().clone(),
            }
    return stats


def _apply_bn_stats(model: nn.Module, bn_dict: dict[str, dict[str, torch.Tensor]]) -> None:
    """Restore BN running stats from a dict produced by ``_collect_bn_stats``."""
    for name, module in model.named_modules():
        if name in bn_dict:
            module.running_mean.copy_(bn_dict[name]["running_mean"])
            module.running_var.copy_(bn_dict[name]["running_var"])
            module.num_batches_tracked.copy_(bn_dict[name]["num_batches_tracked"])


def save_active_bn_stats(model: nn.Module) -> None:
    """Save the current BN buffers under the currently active sparsity key."""
    key = get_active_sparsity(model)
    _bn_buffers[key] = _collect_bn_stats(model)
    n = len(_bn_buffers[key])
    print(f"[BN] Saved BN buffers for key '{key}' ({n} BN layers).")


def restore_bn_stats(model: nn.Module, sparsity_key: str) -> None:
    """Restore BN buffers for *sparsity_key*.  No‑op if no buffers have been saved."""
    if sparsity_key in _bn_buffers:
        _apply_bn_stats(model, _bn_buffers[sparsity_key])
        print(f"[BN] Restored BN buffers for key '{sparsity_key}'.")
    else:
        print(f"[BN] No saved BN buffers for key '{sparsity_key}', using current.")


# ---------------------------------------------------------------------------
# EVALUATING ALL SPARSITY MODES
# ---------------------------------------------------------------------------

def evaluate_all_sparsity_modes(
        model: nn.Module, 
        loader: DataLoader, 
        device: torch.device,
    ) -> dict[str, tuple[float, float]]:
    """
    Evaluates the model on all registered sparsity modes plus dense mode.

    For each sparsity key the corresponding saved BN buffers (if any)
    are restored so that BN running statistics match that mask's activations.

    Args:
        model: The nn.Module to evaluate.
        loader: The DataLoader for evaluation.
        device: The torch.device to use.

    Returns:
        A dictionary mapping sparsity key (e.g., ``"dense"``, ``"1:4"``)
        to a tuple of (average_loss, accuracy).
    """
    results = {}
    
    # Get all registered sparsity keys
    keys = set()
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'nm_masks'):
            keys.update(module.nm_masks.keys())
            break 
    sparsity_keys = sorted(list(keys))
    if 'dense' not in sparsity_keys:
        sparsity_keys.insert(0, 'dense')

    original_active_key = get_active_sparsity(model)

    for key in sparsity_keys:
        set_active_sparsity(model, key)
        restore_bn_stats(model, key)
        _debug_mask_state(model, f"evaluating key='{key}'")
        avg_loss, accuracy = vgg.evaluate(model, loader, device)
        results[key] = (avg_loss, accuracy)
        print(f"[EVAL] Sparsity '{key}': Avg Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
    
    set_active_sparsity(model, original_active_key)
    restore_bn_stats(model, original_active_key)
    return results


def _debug_mask_state(model: nn.Module, tag: str = "") -> None:
    """Print the active mask key and patch state of every module with nm_masks."""
    print(f"\n[DEBUG MASK STATE{(' ' + tag) if tag else ''}]")
    total_conv_linear = sum(
        1 for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, 'weight')
    )
    total_masked = 0
    for name, module in model.named_modules():
        if not hasattr(module, 'nm_masks'):
            continue
        total_masked += 1
        keys = sorted(module.nm_masks.keys())
        active = getattr(module, 'active_mask_key', 'N/A')
        is_patched = (hasattr(module, '_original_forward') and
                      module.forward is not module._original_forward)
        print(f"  {name}: keys={keys} active={active} patched={is_patched}")
    print(f"  [{total_masked}/{total_conv_linear} Conv/Linear layers have nm_masks]")


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    epochs: int = 10,
    N: int = 0,
    M: int = 0,
    verify_frequency: int = 5,
    freeze_params: list[torch.nn.Parameter] | None = None,
    # ------------------------------------------------------------------
    # Optional improvements (backward-compatible defaults)
    # ------------------------------------------------------------------
    lr: float = 0.1,
    scheduler_fn: Callable | None = None,
    label_smoothing: float = 0.0,
    cutout: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #
    # Sparsity injection
    #
    sparsity_key = f"{N}:{M}"
    if N != 0 and M != 0:
        inject_switchable_mask(
            model, 
            sparsity_key, 
            N=N, 
            M=M, 
            conv_axis="out", 
            linear_orientation="row"
        )
        set_active_sparsity(model, sparsity_key)
        print(f"[TRAIN] Sparsity mode '{sparsity_key}' active.")
    else:
        set_active_sparsity(model, "dense")
        print("[TRAIN] Dense mode.")

    #
    # Freeze registration
    #
    if freeze_params:
        for p in freeze_params:
            register_frozen_param(p)
        print(f"[TRAIN] {len(freeze_params)} parameters frozen.")
    else:
        print("[TRAIN] No frozen parameters.")

    #
    # Optimizer / scheduler / dataloader
    #
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    if scheduler_fn is not None:
        scheduler = scheduler_fn(optimizer)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
    train_loader, val_loader = vgg.create_dataloaders("./data", 128, 4, cutout=cutout)

    #
    # Take Snapshot
    #
    snapshot = snapshot_parameters(model)

    #
    # Main training loop
    #
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(outputs, targets)
            loss.backward()

            zero_frozen_gradients(optimizer)
            frozen_snap = _snapshot_frozen_positions()
            optimizer.step()
            _restore_frozen_positions(frozen_snap)

            loss_sum += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += images.size(0)

        scheduler.step()

        val_loss, val_acc = vgg.evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: "
              f"Train Acc {100*correct/total:.2f}% | "
              f"Val Acc {val_acc:.2f}%")

        # 
        # Verify snapshot
        #
        if epoch % verify_frequency == 0:
            verify_after_training(
                model,
                snapshot,
                active_mask_key=sparsity_key,
                frozen_params=freeze_params,
            )
            snapshot = snapshot_parameters(model)

    #
    # Save BN buffers for the active key
    # 
    save_active_bn_stats(model)

    #
    # Clean‑up
    # 
    clear_all_frozen()
    return model


# ---------------------------------------------------------------------------
# FREEZE
# ---------------------------------------------------------------------------

_frozen_params: set[torch.nn.Parameter] = set()
_element_freeze_masks: dict[torch.nn.Parameter, torch.Tensor] = {}

def register_frozen_param(p: torch.nn.Parameter):
    """Mark a Parameter as frozen (no update ever)."""
    _frozen_params.add(p)
    setattr(p, "_frozen", True)

def unregister_frozen_param(p: torch.nn.Parameter):
    """Remove a Parameter from the frozen set."""
    _frozen_params.discard(p)
    if hasattr(p, "_frozen"):
        delattr(p, "_frozen")

def is_frozen(p: torch.nn.Parameter) -> bool:
    return p in _frozen_params or getattr(p, "_frozen", False)

def _snapshot_frozen_positions() -> dict:
    """Cache current values of all frozen positions (element-wise + full param).

    Used before ``optimizer.step()`` so they can be restored afterwards
    (weight decay would otherwise modify them despite zero gradients).
    """
    snap = {}
    for p, freeze_mask in _element_freeze_masks.items():
        if freeze_mask.any():
            snap[id(p)] = p.data[freeze_mask].clone()
    for p in _frozen_params:
        if id(p) not in snap:
            snap[id(p)] = p.data.clone()
    return snap


def _restore_frozen_positions(snap: dict) -> None:
    """Restore frozen positions (element-wise + full param) from a snapshot."""
    for p, freeze_mask in _element_freeze_masks.items():
        pid = id(p)
        if pid in snap:
            p.data[freeze_mask] = snap[pid]
    for p in _frozen_params:
        pid = id(p)
        if pid in snap:
            p.data[:] = snap[pid]


def clear_all_frozen():
    for p in list(_frozen_params):
        if hasattr(p, "_frozen"):
            delattr(p, "_frozen")
    _frozen_params.clear()
    _element_freeze_masks.clear()
    print("[FREEZE] All frozen flags cleared.")

def zero_frozen_gradients(optimizer):
    """Iterate through optimizer's param groups and zero grads of frozen params and element-wise frozen positions."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None and is_frozen(p):
                p.grad.zero_()
    for p, freeze_mask in _element_freeze_masks.items():
        if p.grad is not None:
            p.grad[freeze_mask] = 0.0

def all_trainable_parameters(model: torch.nn.Module):
    """
    Returns a list of (name, parameter) tuples for every parameter that
    requires gradients (i.e. `requires_grad == True`).  These are the
    parameters you could freeze if you wanted to.
    """
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]

def frozen_parameter_names(model: torch.nn.Module):
    """
    Returns a list of the *names* of parameters that have been marked as frozen.
    """
    frozen_names = []
    for name, p in model.named_parameters():
        if is_frozen(p):
            frozen_names.append(name)
    return frozen_names

# ---------------------------------------------------------------------------
# ELEMENT-WISE FREEZE
# ---------------------------------------------------------------------------

def register_element_freeze(param: torch.nn.Parameter, freeze_mask: torch.Tensor):
    """Freeze individual elements of a Parameter (gradient zeroed at those positions).

    Args:
        param: The parameter to freeze elements of.
        freeze_mask: Boolean tensor of the same shape as param. ``True`` = freeze that element.
    """
    freeze_mask = freeze_mask.to(device=param.device, dtype=torch.bool)
    if param in _element_freeze_masks:
        _element_freeze_masks[param] |= freeze_mask
    else:
        _element_freeze_masks[param] = freeze_mask


def clear_element_freeze():
    """Clear all element-level freeze masks."""
    _element_freeze_masks.clear()


# ---------------------------------------------------------------------------
# USER-FACING FREEZE HELPERS
# ---------------------------------------------------------------------------

def freeze_biases(model: torch.nn.Module):
    """Freeze all bias parameters in the model (no gradient updates)."""
    count = 0
    for module in model.modules():
        if hasattr(module, 'bias') and module.bias is not None:
            register_frozen_param(module.bias)
            count += 1
    print(f"[FREEZE] {count} bias parameters frozen.")


def freeze_mask(model: torch.nn.Module, sparsity_key: str):
    """Freeze the weight elements that are *kept* (mask == 1) under the given sparsity mask.

    After calling this, the positions selected by the N:M mask will not receive gradient updates.
    Useful for e.g. fine-tuning only the pruned (inverse-mask) weights.
    """
    count = 0
    for module in model.modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if not hasattr(module, 'nm_masks') or sparsity_key not in module.nm_masks:
            continue
        mask = module.nm_masks[sparsity_key]
        if mask is None:
            continue
        register_element_freeze(module.weight, mask.bool())
        count += 1
    print(f"[FREEZE] {count} modules mask '{sparsity_key}' frozen.")


def freeze_inverse_mask(model: torch.nn.Module, sparsity_key: str):
    """Freeze the weight elements that are *pruned* (mask == 0) under the given sparsity mask.

    After calling this, the positions NOT selected by the N:M mask will not receive gradient updates.
    Useful for e.g. fine-tuning only the sparse (mask) weights.
    """
    count = 0
    for module in model.modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if not hasattr(module, 'nm_masks') or sparsity_key not in module.nm_masks:
            continue
        mask = module.nm_masks[sparsity_key]
        if mask is None:
            continue
        register_element_freeze(module.weight, mask.eq(0))
        count += 1
    print(f"[FREEZE] {count} modules inverse mask '{sparsity_key}' frozen.")


def freeze_batchnorm(model: torch.nn.Module):
    """Freeze all BatchNorm weight and bias parameters in the model."""
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if module.weight is not None:
                register_frozen_param(module.weight)
                count += 1
            if module.bias is not None:
                register_frozen_param(module.bias)
                count += 1
    print(f"[FREEZE] {count} BatchNorm parameters frozen.")


def unfreeze_all():
    """Unfreeze everything: clear all frozen params and element-level freeze masks."""
    clear_all_frozen()
    print("[FREEZE] All parameters unfrozen.")


def freeze_non_masked_layers(model: nn.Module, sparsity_key: str):
    """Freeze the entire weight tensor of Conv/Linear layers not covered by the given mask."""
    count = 0
    for module in model.modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if hasattr(module, 'nm_masks') and sparsity_key in module.nm_masks:
            continue
        if module.weight is not None:
            register_frozen_param(module.weight)
            count += 1
    print(f"[FREEZE] {count} non-masked layers frozen.")


def freeze_all(model: nn.Module):
    """Freeze every single parameter in the model (full-param freeze).

    Useful as a sanity check: training with everything frozen should leave
    every weight unchanged and accuracy should match the pre-training eval.
    """
    count = 0
    for p in model.parameters():
        register_frozen_param(p)
        count += 1
    print(f"[FREEZE] All {count} parameters frozen.")


# ---------------------------------------------------------------------------
# RUNTIME VALIDATION
# ---------------------------------------------------------------------------

def snapshot_parameters(model: nn.Module):
    """
    Returns a dict:
        {
            "params": {name: tensor.clone()},
            "masks":  {module_name: mask_tensor.clone()},
        }
    """
    snap = {"params": {}, "masks": {}}
    for name, param in model.named_parameters():
        snap["params"][name] = param.detach().clone()
    for mod_name, module in model.named_modules():
        if not hasattr(module, "nm_masks"):
            continue
        mask_dict = {}
        for key, mask in module.nm_masks.items():
            if mask is None:
                continue
            mask_dict[key] = mask.detach().clone()
        if mask_dict:
            snap["masks"][mod_name] = mask_dict
    return snap


def verify_after_training(
    model: nn.Module,
    snapshot: dict,
    active_mask_key: str | None = None,
    frozen_params: list[torch.nn.Parameter] | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
):
    """
    Checks that:
      1) All **masked** entries behave like zero and have NOT changed
         relative to the snapshot taken before training.
      2) All **frozen** parameters are exactly the same as the snapshot.

    Parameters
    ----------
    model : nn.Module
        The model after training (or after the checkpoint you want to test).
    snapshot : dict
        The object returned by ``snapshot_parameters`` taken BEFORE training.
    active_mask_key : str 
        Mask key to verify (e.g. "1:4").
    frozen_params : list[torch.nn.Parameter]
        List of parameters that should be frozen
    atol / rtol : tolerance for ``torch.allclose``.
    """
    #
    # Verify frozen parameters
    #
    for name, p in model.named_parameters():
        if is_frozen(p):
            before = snapshot["params"][name]
            if not torch.allclose(p, before, atol=atol, rtol=rtol):
                raise RuntimeError(
                    f"[VERIFY][FREEZE] Parameter '{name}' changed! before : {before}, after  : {p}"
                )
    #
    # Verify element-wise frozen positions
    #
    if not _element_freeze_masks:
        print("[VERIFY] No element-freeze masks registered, skipping element check.")
    else:
        name_to_param = dict(model.named_parameters())
        for name, p in name_to_param.items():
            if p not in _element_freeze_masks:
                continue
            freeze_mask = _element_freeze_masks[p]
            if not freeze_mask.any():
                continue

            before = snapshot["params"][name].to(p.device)
            frozen_pos = freeze_mask
            if not torch.allclose(p.data[frozen_pos], before[frozen_pos], atol=atol, rtol=rtol):
                raise RuntimeError(
                    f"[VERIFY][ELEMENT] Element-frozen positions changed in '{name}'.\n"
                    f"   #frozen entries: {frozen_pos.sum().item()}\n"
                    f"   before[:5] : {before[frozen_pos].view(-1)[:5]}\n"
                    f"   after[:5]  : {p.data[frozen_pos].view(-1)[:5]}"
                )
    print("[VERIFY] All checks passed (frozen + element-freeze).")
    return True


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------

def parameter_snapshot(model: nn.Module) -> dict:
    """Comprehensive snapshot of params, masks, and freeze state for comparison.

    Returns:
        ``{"params": {...}, "masks": {...},
           "frozen_param_names": [...], "element_freeze_masks": {...}}``
    """
    snap = snapshot_parameters(model)
    snap["frozen_param_names"] = frozen_parameter_names(model)
    name_to_param = dict(model.named_parameters())
    snap["element_freeze_masks"] = {}
    for name, p in name_to_param.items():
        if p in _element_freeze_masks:
            snap["element_freeze_masks"][name] = _element_freeze_masks[p].clone()
    return snap


def compare_snapshots(
    model: nn.Module,
    before: dict,
    after: dict,
    sparsity_key: str | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    line_cap: int | None = None,
) -> None:
    """Print a per-element before/after comparison with mask and freeze info.

    Each element is printed on its own line:

        [idx] (before, after, diff, mask, protect)

    where *mask* is 1/0 for weight params (blank for biases) and *protect*
    is ``F`` (frozen) or ``.`` (trainable).

    Args:
        line_cap: If set, only print the first *line_cap* elements per tensor,
                  then append a summary line with the remaining count.
    """
    name_to_param = dict(model.named_parameters())

    print(f"\n{'=' * 70}")
    print("PARAMETER COMPARISON")
    print(f"{'=' * 70}")
    print("  [idx] (before, after, diff, mask, protect)")
    print("  mask:    1 = kept by mask, 0 = pruned, '' = not masked")
    print("  protect: F = frozen, . = trainable")

    device = next(model.parameters()).device

    for name in after["params"]:
        if name not in before["params"]:
            continue

        p = name_to_param[name]
        before_vals = before["params"][name].flatten().to(device)
        after_vals = after["params"][name].flatten().to(device)
        n = before_vals.numel()

        print(f"\n  {name}  ({n} elem)  shape={tuple(p.shape)}")

        # ---- Mask overlay ----
        mask_flat = None
        if sparsity_key is not None and name.endswith(".weight"):
            mod_name = name.rsplit(".weight", 1)[0]
            mod_masks = before.get("masks", {}).get(mod_name, {})
            if sparsity_key in mod_masks:
                mask_flat = mod_masks[sparsity_key].flatten()

        # ---- Freeze protection ----
        is_full_frozen = name in before.get("frozen_param_names", [])
        elem_freeze_flat = None
        if name in before.get("element_freeze_masks", {}):
            elem_freeze_flat = before["element_freeze_masks"][name].flatten()

        # ---- Per-element lines ----
        diff = after_vals - before_vals
        limit = n if line_cap is None else min(line_cap, n)

        protect = ["."] * limit
        if is_full_frozen:
            protect = ["F"] * limit
        if elem_freeze_flat is not None:
            ef = elem_freeze_flat[:limit]
            for i in range(limit):
                if ef[i].item():
                    protect[i] = "F"

        n_changed = 0
        n_protected = 0

        for idx in range(limit):
            b = before_vals[idx].item()
            a = after_vals[idx].item()
            d = diff[idx].item()
            m = int(mask_flat[idx].item()) if mask_flat is not None else ""
            c = protect[idx]
            if c == "F":
                n_protected += 1
            if not torch.isclose(torch.tensor(d), torch.zeros_like(torch.tensor(d)),
                                 atol=atol, rtol=rtol):
                n_changed += 1

            print(f"    [{idx:4d}] ({b:+.4f}, {a:+.4f}, {d:+.4f}, {m}, {c})")

        if line_cap is not None and n > line_cap:
            remaining = diff[line_cap:]
            changed_mask = ~torch.isclose(remaining, torch.zeros_like(remaining),
                                          atol=atol, rtol=rtol)
            n_changed += changed_mask.sum().item()
            if is_full_frozen:
                n_protected += n - line_cap
            elif elem_freeze_flat is not None:
                n_protected += elem_freeze_flat[line_cap:].sum().item()
            print(f"    ... ({n - line_cap} more elements)")

        print(f"    summary: {n_changed}/{n} changed  |  {n_protected} protected  |  {n - n_protected} trained")

    print(f"\n{'=' * 70}\n")


# ---------------------------------------------------------------------------
# CHECKPOINT SAVE / LOAD
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    path: str,
    epoch: int | None = None,
    optimizer: optim.Optimizer | None = None,
    metrics: dict | None = None,
):
    """Save model state, masks, and freezing info to a checkpoint file.

    Args:
        model: The model to save.
        path: File path for the checkpoint.
        epoch: Optional epoch number to store in the checkpoint.
        optimizer: Optional optimizer whose state dict will be saved.
        metrics: Optional dict of metrics to store.
    """
    checkpoint: dict = {
        'model_state_dict': model.state_dict(),
        'masks': {},
        'frozen_param_names': frozen_parameter_names(model),
        'element_freeze_masks': {},
        'bn_buffers': {},
    }

    for mname, module in model.named_modules():
        if not hasattr(module, 'nm_masks'):
            continue
        saved = {k: v.cpu() for k, v in module.nm_masks.items() if v is not None}
        if saved:
            checkpoint['masks'][mname] = saved

    name_to_param = dict(model.named_parameters())
    for name, p in name_to_param.items():
        if p in _element_freeze_masks:
            checkpoint['element_freeze_masks'][name] = _element_freeze_masks[p].cpu()

    # Persist per-key BN buffers (cpu)
    for skey, mdict in _bn_buffers.items():
        checkpoint['bn_buffers'][skey] = {
            mname: {bk: bv.cpu() for bk, bv in bd.items()}
            for mname, bd in mdict.items()
        }

    if epoch is not None:
        checkpoint['epoch'] = epoch
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, path)
    print(f"[SAVE] Checkpoint saved to {path}")
    if metrics and 'eval' in metrics:
        print(f"  eval summary:")
        for key, (loss, acc) in metrics['eval'].items():
            print(f"    {key:<10}  loss={loss:.4f}  acc={acc:.2f}%")


def load_checkpoint(
    model: nn.Module,
    path: str,
    strict: bool = True,
    device: torch.device | None = None,
) -> dict:
    """Load model state, masks, and freezing info from a checkpoint file.

    Args:
        model: The model to load into (should have the same architecture).
        path: File path of the checkpoint.
        strict: Whether to enforce strict key matching in ``load_state_dict``.
        device: Device to load tensors onto (default: model's current device).

    Returns:
        The checkpoint dict (useful for accessing epoch, metrics, etc.).
    """
    map_location = device
    checkpoint = torch.load(path, weights_only=True, map_location=map_location)

    # Restore nm_masks and patch forwards
    for mname, module in model.named_modules():
        saved_masks = checkpoint.get('masks', {}).get(mname)
        if saved_masks is None:
            continue
        if not hasattr(module, 'nm_masks'):
            module.nm_masks = {'dense': None}
        for key, mask in saved_masks.items():
            module.nm_masks[key] = mask.to(module.weight.device)
        _patch_forward_for_masks(module)

    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    # Restore frozen info
    clear_all_frozen()
    name_to_param = dict(model.named_parameters())
    for name in checkpoint.get('frozen_param_names', []):
        if name in name_to_param:
            register_frozen_param(name_to_param[name])

    clear_element_freeze()
    for name, mask in checkpoint.get('element_freeze_masks', {}).items():
        if name in name_to_param:
            _element_freeze_masks[name_to_param[name]] = mask.bool()

    # Restore per-key BN buffers
    _bn_buffers.clear()
    for skey, mdict in checkpoint.get('bn_buffers', {}).items():
        _bn_buffers[skey] = mdict  # kept on cpu; moved to active device at restore time
    if _bn_buffers:
        print(f"  Restored BN buffers for keys: {list(_bn_buffers.keys())}")
    else:
        print(f"  [BN] No BN buffers found in checkpoint")

    print(f"[LOAD] Checkpoint loaded from {path}")
    saved_metrics = checkpoint.get('metrics', {})
    if 'eval' in saved_metrics:
        print(f"  original eval:")
        for key, (loss, acc) in saved_metrics['eval'].items():
            print(f"    {key:<10}  loss={loss:.4f}  acc={acc:.2f}%")
    return checkpoint

