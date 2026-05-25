import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


@torch.no_grad()
def _nm_mask_2d_rows(x2d: torch.Tensor, N: int, M: int) -> torch.Tensor:
    """
    Vectorized N:M masking along the LAST dim of a 2D tensor (row-wise).
    Keeps N largest-by-magnitude within every contiguous group of size M.
    Returns a 0/1 mask with same shape/device/dtype as x2d.
    """
    if not (isinstance(N, int) and isinstance(M, int) and 0 <= N <= M and M > 0):
        raise ValueError("Require integers with 0 <= N <= M and M > 0.")
    R, C = x2d.shape
    if C == 0 or N == 0:
        return torch.zeros_like(x2d)

    x_abs = x2d.abs()
    pad = (M - (C % M)) % M
    if pad:
        pad_vals = torch.full(
            (R, pad), float("-inf"), device=x2d.device, dtype=x_abs.dtype
        )
        x_abs_p = torch.cat([x_abs, pad_vals], dim=1)
    else:
        x_abs_p = x_abs

    R, Cpad = x_abs_p.shape
    G = Cpad // M
    xg = x_abs_p.view(R, G, M)  # [R, G, M]
    topk_idx = torch.topk(xg, N, dim=2, largest=True).indices
    mg = torch.zeros_like(xg)
    mg.scatter_(2, topk_idx, 1)
    m_flat = mg.view(R, Cpad)
    if pad:
        m_flat = m_flat[:, :C]
    return m_flat.type_as(x2d)


@torch.no_grad()
def _nm_mask_conv2d_axis(
    weight: torch.Tensor, N: int, M: int, axis: str
) -> torch.Tensor:
    """
    weight: [a=out_c, b=in_c, c=kH, d=kW]

    axis='in'  -> N:M only along b (input channels).
                 We create a 2D view [a*c*d, b], mask blocks along 'b'.

    axis='out' -> N:M only along a (output channels).
                 We create a 2D view [b*c*d, a], mask blocks along 'a'.
    """
    a, b, c, d = weight.shape

    if axis == "in":
        # View as rows of (a*c*d), columns=b (mask along last dim)
        W2 = weight.permute(0, 2, 3, 1).reshape(a * c * d, b)  # [a*c*d, b]
        M2 = _nm_mask_2d_rows(W2, N, M)  # same shape
        mask = M2.reshape(a, c, d, b).permute(0, 3, 1, 2).contiguous()  # -> [a,b,c,d]
        return mask.type_as(weight)

    elif axis == "out":
        # View as rows of (b*c*d), columns=a (mask along last dim)
        W2 = weight.permute(1, 2, 3, 0).reshape(b * c * d, a)  # [b*c*d, a]
        M2 = _nm_mask_2d_rows(W2, N, M)
        mask = M2.reshape(b, c, d, a).permute(3, 0, 1, 2).contiguous()  # -> [a,b,c,d]
        return mask.type_as(weight)

    else:
        raise ValueError("axis must be 'in' or 'out'.")


@torch.no_grad()
def _nm_mask_linear(weight: torch.Tensor, N: int, M: int, orientation: str):
    """
    For Linear:
      - orientation='row' : N:M across input features (columns) per output row.
      - orientation='col' : N:M across output features (rows) per input column.
    """
    if orientation == "row":
        return _nm_mask_2d_rows(weight, N, M)
    elif orientation == "col":
        return _nm_mask_2d_rows(weight.transpose(0, 1), N, M).transpose(0, 1)
    else:
        raise ValueError("orientation must be 'row' or 'col'.")


@torch.no_grad()
def prune_nm(
    model: nn.Module,
    N: int,
    M: int,
    conv_axis: str = "row",
    linear_orientation: str = "row",
):
    """
    Apply N:M pruning to a model using PyTorch's CustomFromMask.

    conv_axis: 'in' -> N:M along input channels for Conv2d.
               'out' -> N:M along output channels for Conv2d.

    linear_orientation: 'row' or 'col'.
    """
    if not (isinstance(N, int) and isinstance(M, int) and 0 <= N <= M and M > 0):
        raise ValueError("Require integers with 0 <= N <= M and M > 0.")
    if conv_axis not in ("in", "out"):
        raise ValueError("conv_axis must be 'in' or 'out'.")
    if linear_orientation not in ("row", "col"):
        raise ValueError("linear_orientation must be 'row' or 'col'.")

    masks = {}

    conv_linear = [
        m for m in model.modules()
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)) and hasattr(m, "weight")
    ]
    first_last = set(conv_linear[:1] + conv_linear[-1:]) if conv_linear else set()

    for module in model.modules():
        if not ((isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and hasattr(module, "weight")):
            continue
        if module in first_last:
            continue

        w = module.weight.detach()
        if isinstance(module, nn.Conv2d):
            mask = _nm_mask_conv2d_axis(w, N, M, axis=conv_axis)
        else:
            mask = _nm_mask_linear(w, N, M, orientation=linear_orientation)

        prune.CustomFromMask.apply(module, name="weight", mask=mask)
        masks[module] = mask

    for module in masks:
        if hasattr(module, "weight_mask") and hasattr(module, "weight_orig"):
            mask = module.weight_mask
            module.weight_orig.register_hook(lambda g, mask=mask: g * mask)

    def _enforce_mask(_mod, _inp):
        if hasattr(_mod, "weight_mask") and hasattr(_mod, "weight_orig"):
            with torch.no_grad():
                _mod.weight_orig.mul_(_mod.weight_mask)

    for module in masks:
        module.register_forward_pre_hook(_enforce_mask)

    return masks

