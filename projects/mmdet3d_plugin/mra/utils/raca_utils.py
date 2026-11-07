"""RaCA: Range- and Class-Aware Critical Voxel Alignment -- pure-torch helpers for components C2, C3 and C4.

C1  DepthScanBlock                 depth-axis Scan block (masked self-attention, near-to-far cascade mask, margin)
    scan_loss_depth               depth-axis Scan loss (back-to-front cumulative mean of logits and labels)
C2  stratified_critical_voxels    one shared set of critical voxel locations for both resolution grids
C3  directed_alignment            co-located, directed (stop-gradient) KL between the two grids
C4  coarse_labels_rare_preserving rare-class-preserving 2x2x2 pooling of the coarse supervision
    pool_proposal                 max-pooled coarse seed proposal (the only inference-time change)

Authors: Farquleet Farhat Gondal, Daniyal Imran (extension of MRA, Yang & Peng, IEEE TIP 2026).

Written for torch 1.9 (MRA's environment): no torch.isin, no scalar arguments to torch.where.
Class ids follow SemanticKITTI SSC: 0 empty, 1..19 classes, 255 unknown. Grids are (X forward, Y lateral, Z height).
"""
import torch
import torch.nn as nn
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


# ----------------------------------------------------------------------------------------------------------------
# C1: depth-axis scan, after ScanSSC (Bae, Ha and Kim, CVPR 2025, arXiv 2411.16129), restricted to the depth axis.
# Module = ScanSSC's Scan block (Algorithm G.1 of the paper) applied along the forward (X) axis only:
#   pre-norm masked self-attention + 2-layer FFN, residual connections, cascading mask so that a voxel attends to
#   itself and to the voxels nearer to the camera, with the nearest `margin` share of every line unmasked for all.
# Loss = ScanSSC's Scan loss (Algorithm G.2) for the depth axis: cumulative mean of logits and of one-hot labels
#   from the far end of each line to the near end, cross-entropy between them.
# Differences from ScanSSC, on purpose: only the depth axis (no width/height blocks, no spatial mixing network and
#   no tri-axis softmax fusion); unknown voxels (255) are excluded from the label average; the last projection of
#   the attention and of the FFN can be zero-initialised so a warm-started model starts exactly as MRA.
# ----------------------------------------------------------------------------------------------------------------

def depth_cascade_mask(length, margin=0.5, device=None):
    """(L, L) bool mask for nn.MultiheadAttention (True = may NOT attend). Query i may attend keys j <= i
    (near-to-far cascade) and every key in the nearest `margin` share of the line (ScanSSC: `[:, :X//2] = False`)."""
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
    mask[:, :int(round(length * margin))] = False
    return mask


class DepthScanBlock(nn.Module):
    """One ScanSSC Scan block along the depth axis of a (B, C, X, Y, Z) volume; every (y, z) line is a sequence."""

    def __init__(self, dim, seq_len, heads=4, margin=0.5, pos_embed=False, zero_init=True):
        super().__init__()
        self.margin = margin
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, dim * 2)
        self.act = nn.ReLU()
        self.ff2 = nn.Linear(dim * 2, dim)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim)) if pos_embed else None
        if zero_init:                                  # block is the identity at initialisation
            nn.init.zeros_(self.attn.out_proj.weight); nn.init.zeros_(self.attn.out_proj.bias)
            nn.init.zeros_(self.ff2.weight); nn.init.zeros_(self.ff2.bias)
        self._mask = {}

    def mask(self, length, device):
        key = (length, str(device))
        if key not in self._mask:
            self._mask[key] = depth_cascade_mask(length, self.margin, device)
        return self._mask[key]

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        seq = x.permute(0, 3, 4, 2, 1).reshape(B * Y * Z, X, C)      # one sequence per (y, z) line, near -> far
        if self.pos is not None:
            seq = seq + self.pos[:, :X]
        h = self.norm1(seq)
        seq = seq + self.attn(h, h, h, attn_mask=self.mask(X, x.device), need_weights=False)[0]
        seq = seq + self.ff2(self.act(self.ff1(self.norm2(seq))))
        return seq.reshape(B, Y, Z, X, C).permute(0, 4, 3, 1, 2).contiguous()


def scan_loss_depth(logits, target, ignore_index=255):
    """ScanSSC Scan loss along the depth axis. logits (B, C, X, Y, Z), target (B, X, Y, Z) with `ignore_index`.
    Logits and one-hot labels are cumulatively averaged from the far end of each line (back to front); labels are
    averaged over the labelled voxels seen so far, and positions that have not yet seen a labelled voxel are skipped."""
    B, C, X = logits.shape[:3]
    valid = target != ignore_index
    t = torch.where(valid, target.long(), torch.zeros_like(target, dtype=torch.long))
    onehot = F.one_hot(t, C).permute(0, 4, 1, 2, 3).to(logits.dtype) * valid.unsqueeze(1).to(logits.dtype)
    count = torch.arange(1, X + 1, device=logits.device, dtype=logits.dtype).view(1, 1, X, 1, 1)
    cum_logit = torch.cumsum(logits.flip(2), dim=2) / count                     # back-to-front mean of logits
    cum_t = torch.cumsum(onehot.flip(2), dim=2)
    n_valid = cum_t.sum(1, keepdim=True)
    cum_t = cum_t / n_valid.clamp(min=1.0)                                      # back-to-front class distribution
    loss = -(cum_t * F.log_softmax(cum_logit, dim=1)).sum(1)                    # soft-target cross-entropy
    keep = n_valid.squeeze(1) > 0
    if not bool(keep.any()):
        return logits.sum() * 0.0
    return loss[keep].mean()
