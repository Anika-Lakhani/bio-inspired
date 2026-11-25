# Paper's models (classes loaded verbatim from internal-code.ipynb), with the two bugs fixed in the harness:
#  - noise added in pixel space before normalization (no clamp of normalized values)
#  - ablation flags passed explicitly so each variant is really what its name says
import json, sys, time, torch, torch.nn as nn, torchvision, torchvision.transforms as T, numpy as np
from torch.utils.data import DataLoader, Subset

nb = json.load(open(sys.argv[1]))
ns = {"__name__": "nb"}
exec("".join(nb["cells"][0]["source"]), ns)
ControlCNN, ExperimentalCNN = ns["ControlCNN"], ns["ExperimentalCNN"]
import torch.nn.functional as F

class ZeroSumInhibition(nn.Module):
    """Center-surround inhibition within each channel: a per-channel DoG kernel kept at zero sum
    (excitatory center, inhibitory surround), whose strength alpha starts at 0 so the model
    begins identical to the pretrained network."""
    def __init__(self, channels, k=5):
        super().__init__()
        g = torch.linspace(-(k//2), k//2, k); d = g[:, None]**2 + g[None, :]**2
        center, surround = torch.exp(-d/(2*(k/6)**2)), torch.exp(-d/(2*(k/3)**2))
        dog = center/center.sum() - surround/surround.sum()
        self.weight = nn.Parameter(dog.repeat(channels, 1, 1, 1))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.channels, self.pad = channels, k//2
    def forward(self, x):
        w = self.weight - self.weight.mean(dim=(2, 3), keepdim=True)  # stays zero-sum while training
        return x + self.alpha * F.conv2d(x, w, padding=self.pad, groups=self.channels)

def with_new_inhibition(m):
    m.inhibition = ZeroSumInhibition(m.actual_channels)
    return m

dev = torch.device("mps")
norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
base = T.Compose([T.Resize((224,224)), T.ToTensor()])
train_full = torchvision.datasets.CIFAR10("./data", True, transform=T.Compose([base, norm]))
test_px = Subset(torchvision.datasets.CIFAR10("./data", False, transform=base), range(5000))

# Control is unchanged from iteration 1 (same seeds, subsets and corrupted test images), so its
# results are loaded from there. Each config returns (model, its inhibition layer, force alpha >= 0).
def late(nonneg):
    m = with_new_inhibition(ExperimentalCNN(use_inhib=True, use_attn=False))
    return m, m.inhibition, nonneg
def early(nonneg):
    # Right after the stem convolution (16 channels at 112x112), the closest stage to the retina.
    m = ExperimentalCNN(use_inhib=False, use_attn=False)
    inhib = ZeroSumInhibition(16)
    m.features = nn.Sequential(m.features[0], inhib, *m.features[1:])
    return m, inhib, nonneg
configs = {
    "Late, a free": lambda: late(False),
    "Late, a>=0": lambda: late(True),
    "Early, a>=0": lambda: early(True),
    "Early, a free": lambda: early(False),
}
prev = json.load(open(sys.argv[2]))
def noise(s):
    return lambda x, g: norm(torch.clamp(x + torch.randn(x.shape, generator=g)*s, 0, 1))
def occlude(x, g):
    x = norm(x).clone()
    for i in range(len(x)):
        r, c = torch.randint(0, 224-50, (2,), generator=g).tolist(); x[i, :, r:r+50, c:c+50] = 0
    return x
conds = {"clean": lambda x, g: norm(x), "s=0.01": noise(0.01), "s=0.03": noise(0.03),
         "s=0.05": noise(0.05), "s=0.1": noise(0.1), "occluded": occlude}

SEEDS = [0]
res = {m: {c: [] for c in conds} for m in configs}
for seed in SEEDS:
    idx = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed))[:20000].tolist()
    for name, make in configs.items():
        torch.manual_seed(seed)
        m, inhib, nonneg = make(); m.to(dev)
        # New parameters (inhibition kernel, its strength, the attention/inhibition mix) get a higher
        # learning rate; at 1e-4, alpha could move at most ~0.03 in one epoch and never engage.
        new_ids = {id(p) for p in inhib.parameters()}
        new = [p for p in m.parameters() if id(p) in new_ids]
        rest = [p for p in m.parameters() if id(p) not in new_ids]
        opt = torch.optim.AdamW([{"params": rest, "lr": 1e-4}, {"params": new, "lr": 1e-2}],
                                weight_decay=1e-5)
        m.train(); t0 = time.time()
        for x, y in DataLoader(Subset(train_full, idx), 64, shuffle=True,
                               generator=torch.Generator().manual_seed(seed)):
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            nn.functional.cross_entropy(m(x), y).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if nonneg:  # projected gradient: alpha can sit at 0 but never go negative
                with torch.no_grad(): inhib.alpha.clamp_(min=0)
        with torch.no_grad():
            w = inhib.weight - inhib.weight.mean(dim=(2, 3), keepdim=True)
            k = (inhib.alpha * w)[:, 0].cpu()  # effective kernel per channel
            c = k.shape[-1] // 2
            center = k[:, c, c]
            g = torch.linspace(-c, c, 2*c+1); d = g[:, None]**2 + g[None, :]**2
            ce, su = torch.exp(-d/(2*((2*c+1)/6)**2)), torch.exp(-d/(2*((2*c+1)/3)**2))
            dog = (ce/ce.sum() - su/su.sum()).flatten()
            cos = torch.nn.functional.cosine_similarity(k.flatten(1), dog[None], dim=1)
            mag = k.flatten(1).norm(dim=1)
        print(f"{name:14s} alpha {inhib.alpha.item():+.3f} | effective center mean {center.mean():+.4f} | "
              f"channels center>0 {(center > 0).float().mean():.0%} | cosine to DoG mean {cos.mean():+.2f}, "
              f"channels >0 {(cos > 0).float().mean():.0%} | kernel norm mean {mag.mean():.3f}", flush=True)
