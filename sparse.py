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

