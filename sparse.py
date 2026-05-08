import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.optim import lr_scheduler
from vgg import create_dataloaders, evaluate


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

# ---------------------------------------------------------------------------
# SWITCHABLE MASKS 
# ---------------------------------------------------------------------------

@torch.no_grad()
def inject_switchable_masks(
    model: nn.Module,
    N: int,
    M: int,
    conv_axis: str = "out",
    linear_orientation: str = "row",
):
    """
    Inject N:M switchable sparsity masks into all Conv2d / Linear layers of
    'model', skipping the very first and very last such layers.

    For each eligible layer this function:

    1. Computes the N:M mask from the current weights and registers it as a
       persistent buffer (``module.nm_mask``).
    2. Sets ``module.use_sparse = False`` (dense mode by default).
    3. Patches ``module.forward`` so that when ``use_sparse`` is
       ``True`` the weight is multiplied by the mask before the operation.

    Calling this function more than once on the same model is safe.

    Args:
        model:               The target ``nn.Module``.
        N:                   Number of weights to keep per group.
        M:                   Group size.
        conv_axis:           Axis for Conv2d mask (``"out"`` or ``"in"``).
        linear_orientation:  Orientation for Linear mask (``"row"`` or
                             ``"col"``).
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

        if hasattr(module, 'nm_mask'):
            continue

        # 1. Generate the N:M Mask
        w = module.weight.detach()
        if isinstance(module, nn.Conv2d):
            mask = _nm_mask_conv2d_axis(w, N, M, axis=conv_axis)
        else:
            mask = _nm_mask_linear(w, N, M, orientation=linear_orientation)

        # 2. Register the mask as a buffer (saved but not trained)
        module.register_buffer('nm_mask', mask)
        module.use_sparse = False # Default state

        # 3. Dynamic Forward Patching
        if not hasattr(module, "_original_forward"):
            module._original_forward = module.forward
        
        def _make_forward(m):
            def _forward(input):
                if getattr(m, 'use_sparse', False):
                    # APPLY MASK: Sparse Mode (1:4)
                    masked_weight = m.weight * m.nm_mask
                    
                    if isinstance(m, nn.Linear):
                        return F.linear(input, masked_weight, m.bias)
                    else:
                        return F.conv2d(input, masked_weight, m.bias, 
                                        m.stride, m.padding, m.dilation, m.groups)
                else:
                    # DENSE: 1:1 Mode
                    return m._original_forward(input)
            return _forward

        module.forward = _make_forward(module)



