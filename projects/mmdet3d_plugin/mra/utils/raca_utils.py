"""RaCA: Range- and Class-Aware Critical Voxel Alignment -- pure-torch helpers for components C2, C3 and C4.

C2  stratified_critical_voxels    one shared set of critical voxel locations for both resolution grids
C3  directed_alignment            co-located, directed (stop-gradient) KL between the two grids
C4  coarse_labels_rare_preserving rare-class-preserving 2x2x2 pooling of the coarse supervision
    pool_proposal                 max-pooled coarse seed proposal (the only inference-time change)

Authors: Farquleet Farhat Gondal, Daniyal Imran (extension of MRA, Yang & Peng, IEEE TIP 2026).

Written for torch 1.9 (MRA's environment): no torch.isin, no scalar arguments to torch.where.
Class ids follow SemanticKITTI SSC: 0 empty, 1..19 classes, 255 unknown. Grids are (X forward, Y lateral, Z height).
"""
import torch
import torch.nn.functional as F

RARE = (2, 3, 6, 7, 8)                                 # bicycle, motorcycle, person, bicyclist, motorcyclist
FOREGROUND = (1, 2, 3, 4, 5, 6, 7, 8, 14, 18, 19)     # MRA's own "thing" group (get_csa, c1[0])
STRATA = ("rare", "foreground", "far", "rest", "random")


def _lut(ids, device):
    t = torch.zeros(256, dtype=torch.bool, device=device)
    t[list(ids)] = True
    return t


def _pick(mask, n):
    """Up to n distinct flat indices where mask is True (uniformly at random)."""
    idx = mask.nonzero(as_tuple=False).squeeze(1)
    if n <= 0 or idx.numel() == 0:
        return idx[:0]
    if idx.numel() <= n:
        return idx
    return idx[torch.randperm(idx.numel(), device=idx.device)[:n]]


@torch.no_grad()
def stratified_critical_voxels(target, N=4096, shares=(0.25, 0.25, 0.40, 0.10), rare_share=0.5, far_x=128):
    """C2: one shared set of N voxel-centre locations for both grids.

    target: (1, X, Y, Z) labels at full resolution (256 x 256 x 32).
    Strata: foreground (with a rare sub-quota of rare_share of the foreground budget when the frame has rare voxels),
    far background (x >= far_x, i.e. beyond 25.6 m), rest, and uniform random. A stratum with too few voxels passes
    its unused budget on: rare -> foreground -> far -> rest -> random.
    Returns coords (1, N, 3) in MRA's normalised grid_sample order (z, y, x) and the stratum id (1, N) of each voxel.
    """
    assert target.shape[0] == 1, "MRA trains with one sample per GPU"
    _, X, Y, Z = target.shape
    dev = target.device
    t = target[0].long().reshape(-1)
    valid = t != 255
    fg = _lut(FOREGROUND, dev)[t]
    rare = _lut(RARE, dev)[t]
    xi = torch.arange(X, device=dev).view(X, 1, 1).expand(X, Y, Z).reshape(-1)
    far = valid & ~fg & (xi >= far_x)
    rest = valid & ~fg & (xi < far_x)

    n_fg = int(round(N * shares[0])); n_far = int(round(N * shares[1])); n_rest = int(round(N * shares[2]))
    n_rare = int(n_fg * rare_share) if bool(rare.any()) else 0
    chosen, tags = [], []

    def take(mask, n, tag):
        sel = _pick(mask, n)
        chosen.append(sel); tags.append(torch.full_like(sel, tag))
        return n - sel.numel()

    left = take(rare, n_rare, 0)
    left = take(fg & ~rare, n_fg - n_rare + left, 1)
    left = take(far, n_far + left, 2)
    take(rest, n_rest + left, 3)
    n_rand = N - sum(c.numel() for c in chosen)
    chosen.append(torch.randint(0, t.numel(), (n_rand,), device=dev)); tags.append(torch.full((n_rand,), 4, dtype=torch.long, device=dev))

    flat = torch.cat(chosen); stratum = torch.cat(tags)
    x = torch.div(flat, Y * Z, rounding_mode='floor'); y = torch.div(flat, Z, rounding_mode='floor') % Y; z = flat % Z
    # grid_sample on a (B, C, X, Y, Z) volume reads the last coordinate as X and the first as Z
    coords = torch.stack([(z.float() + 0.5) / Z, (y.float() + 0.5) / Y, (x.float() + 0.5) / X], dim=1)
    return coords.unsqueeze(0), stratum.unsqueeze(0)


@torch.no_grad()
def coarse_labels_rare_preserving(target):
    """C4: label of each 2x2x2 block = a rare class if any voxel in the block has one (the most frequent rare class),
    otherwise the majority label of the labelled voxels (empty included), otherwise 255. Same dtype as target."""
    B, X, Y, Z = target.shape
    t = target.long().reshape(B, X // 2, 2, Y // 2, 2, Z // 2, 2).permute(0, 1, 3, 5, 2, 4, 6).reshape(B, X // 2, Y // 2, Z // 2, 8)
    counts = torch.stack([(t == c).sum(-1) for c in range(20)], dim=-1)        # (B, x, y, z, 20)
    majority = counts.argmax(-1)
    unlabelled = counts.sum(-1) == 0
    rare_ids = torch.tensor(RARE, device=target.device)
    rare_counts = counts[..., list(RARE)]
    has_rare = rare_counts.sum(-1) > 0
    rare_label = rare_ids[rare_counts.argmax(-1)]
    out = torch.where(has_rare, rare_label, torch.where(unlabelled, torch.full_like(majority, 255), majority))
    return out.to(target.dtype)


def pool_proposal(proposal):
    """C4: coarse seed proposal = max over each 2x2x2 block (numpy, shape (B, X, Y, Z))."""
    B, X, Y, Z = proposal.shape
    return proposal.reshape(B, X // 2, 2, Y // 2, 2, Z // 2, 2).max(axis=(2, 4, 6))


def directed_alignment(logits_fine, logits_coarse, stratum):
    """C3: per-voxel KL over classes between the two grids at the same locations, with a stop-gradient on the leader.
    The fine grid leads on the rare and foreground strata; elsewhere the grid with lower predictive entropy leads.
    logits_*: (1, C, N); stratum: (1, N)."""
    lf = F.log_softmax(logits_fine, dim=1)
    lc = F.log_softmax(logits_coarse, dim=1)
    ent_f = -(lf.exp() * lf).sum(1)
    ent_c = -(lc.exp() * lc).sum(1)
    fine_leads = (stratum <= 1) | (ent_f.detach() <= ent_c.detach())
    kl_fine_leads = (lf.detach().exp() * (lf.detach() - lc)).sum(1)    # coarse follows fine
    kl_coarse_leads = (lc.detach().exp() * (lc.detach() - lf)).sum(1)  # fine follows coarse
    return torch.where(fine_leads, kl_fine_leads, kl_coarse_leads).mean(), fine_leads.float().mean()
