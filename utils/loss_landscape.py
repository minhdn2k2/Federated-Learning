# loss_landscape.py
import math, copy, torch, numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector, vector_to_parameters

@torch.no_grad()
def _make_directions_like(model, seed=None):
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    dirs = []
    for p in model.parameters():
        if p.requires_grad:
            dirs.append(torch.randn_like(p))
        else:
            dirs.append(torch.zeros_like(p))
    return dirs

@torch.no_grad()
def _filterwise_normalize(dirs, ref_params):
    """
    Filter-wise normalization: for each param tensor:
      - conv/linear weight: scale direction so that ||d_ij...|| matches ||w_ij...|| per-filter
      - bias/norm params: scale by global ratio of norms
    This makes directions scale-invariant wrt parameter magnitudes.
    """
    normed = []
    for d, w in zip(dirs, ref_params):
        if d.numel() == 0:
            normed.append(d)
            continue
        # Try to detect "filter" dimension: conv [out,in,kH,kW], linear [out,in]
        if d.dim() >= 2:  # treat dim0 as "out" filters
            d_ = d.clone()
            w_ = w.clone()
            # reshape to [out, -1]
            d_flat = d_.view(d_.shape[0], -1)
            w_flat = w_.view(w_.shape[0], -1)
            d_norm = d_flat.norm(p=2, dim=1, keepdim=True) + 1e-12
            w_norm = w_flat.norm(p=2, dim=1, keepdim=True)
            scaled = (d_flat / d_norm) * w_norm
            normed.append(scaled.view_as(d_))
        else:
            # biases, norm weights; match global l2
            d_norm = d.norm(p=2) + 1e-12
            w_norm = w.norm(p=2)
            normed.append(d * (w_norm / d_norm))
    return normed

@torch.no_grad()
def _add_directions_to_params(model, base_vec, d1_vec, d2_vec, a, b):
    vec = base_vec + a * d1_vec + b * d2_vec
    vector_to_parameters(vec, [p for p in model.parameters() if p.requires_grad])

def evaluate_loss(model, dataloader, loss_fn, device):
    model.eval()
    tot_loss, tot_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device).float()
            yb = yb.to(device).long().view(-1)
            out = model(xb)
            loss = loss_fn(out, yb)  # expect reduction='mean'
            bs = yb.numel()
            tot_loss += loss.item() * bs
            tot_n += bs
    return tot_loss / max(tot_n, 1)

def compute_landscape_2d(model, val_loader, loss_fn, device,
                         radius_alpha=1.0, radius_beta=1.0, grid=25, seed=0,
                         subsample_batches=None):
    """
    Returns: alphas, betas (1D), loss_grid [grid x grid]
    """
    model = model.to(device)
    # Snapshot base weights
    base_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    base_vec = parameters_to_vector([p for p in model.parameters() if p.requires_grad]).detach().clone()

    # Random directions -> filter-wise normalize -> vectorize
    d1 = _make_directions_like(model, seed=seed)
    d2 = _make_directions_like(model, seed=seed+1 if seed is not None else None)
    d1 = _filterwise_normalize(d1, base_params)
    d2 = _filterwise_normalize(d2, base_params)
    d1_vec = parameters_to_vector(d1).to(device)
    d2_vec = parameters_to_vector(d2).to(device)

    # Optional: subsample validation for speed
    if subsample_batches is not None and subsample_batches > 0:
        from itertools import islice
        val_iter = iter(val_loader)
        small_list = list(islice(val_iter, subsample_batches))
        class SmallLoader:
            def __iter__(self): return iter(small_list)
        eval_loader = SmallLoader()
    else:
        eval_loader = val_loader

    # Build grid
    alphas = torch.linspace(-radius_alpha, radius_alpha, grid)
    betas  = torch.linspace(-radius_beta,  radius_beta,  grid)
    Z = np.zeros((grid, grid), dtype=np.float64)

    # Evaluate
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            _add_directions_to_params(model, base_vec.to(device), d1_vec, d2_vec, a.item(), b.item())
            Z[i, j] = evaluate_loss(model, eval_loader, loss_fn, device)

    # Restore weights
    vector_to_parameters(base_vec.to(device), [p for p in model.parameters() if p.requires_grad])
    return alphas.cpu().numpy(), betas.cpu().numpy(), Z

def plot_landscape(alphas, betas, Z, title="Loss landscape", elev=35, azim=-60):
    A, B = np.meshgrid(alphas, betas, indexing='ij')
    # 2D contour
    plt.figure(figsize=(6,5))
    cs = plt.contour(A, B, Z, levels=20)
    plt.clabel(cs, inline=1, fontsize=8)
    plt.xlabel("alpha (dir-1)")
    plt.ylabel("beta (dir-2)")
    plt.title(title + " - Contour")
    plt.tight_layout()

    # 3D surface
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A, B, Z, linewidth=0, antialiased=True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_zlabel("loss")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title + " - Surface")
    plt.tight_layout()
    plt.show()
