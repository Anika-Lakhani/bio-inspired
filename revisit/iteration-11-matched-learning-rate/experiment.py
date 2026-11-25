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

# Confirmation run: only the Control and the iteration-3 winner, on 10 fresh seeds (3-12) that
# were not used to choose the design. Each config returns (model, inhibition layer or None, alpha >= 0).
class PlainFilter(nn.Module):
    """Non-biological control: same spot, size, strength-starts-at-0 and learning rate as the
    early inhibition layer, but a randomly initialized 5x5 depthwise kernel with no zero-sum or
    sign constraint, so training can learn any local filter (blur, sharpen, edge, ...)."""
    def __init__(self, channels, k=5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(channels, 1, k, k) * 0.1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.channels, self.pad = channels, k//2
    def forward(self, x):
        return x + self.alpha * F.conv2d(x, self.weight, padding=self.pad, groups=self.channels)

class FrozenInhibition(nn.Module):
    """Fixed biological prior (VOneNet-style): the zero-sum center-surround kernel and its strength
    are set by hand and never trained, so learning on clean images can't reshape or switch them off."""
    def __init__(self, channels, alpha, k=5):
        super().__init__()
        g = torch.linspace(-(k//2), k//2, k); d = g[:, None]**2 + g[None, :]**2
        center, surround = torch.exp(-d/(2*(k/6)**2)), torch.exp(-d/(2*(k/3)**2))
        dog = center/center.sum() - surround/surround.sum()
        self.register_buffer("weight", dog.repeat(channels, 1, 1, 1))
        self.register_buffer("alpha", torch.tensor([float(alpha)]))
        self.channels, self.pad = channels, k//2
    def forward(self, x):
        return x + self.alpha * F.conv2d(x, self.weight, padding=self.pad, groups=self.channels)

class NeuralNoise(nn.Module):
    """VOneNet-style stochasticity: Poisson-like noise whose variance grows with response magnitude,
    applied during training AND testing (as in VOneNet), right after the early stage."""
    def __init__(self, scale=0.3):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x + self.scale * torch.sqrt(x.abs() + 1e-6) * torch.randn_like(x)

def early_stack(*layers):
    m = ExperimentalCNN(use_inhib=False, use_attn=False)
    m.features = nn.Sequential(m.features[0], *layers, *m.features[1:])
    return m

def frozen_noise(sc):
    return early_stack(FrozenInhibition(16, 0.5), NeuralNoise(sc)), None, False

def plain_noise(sc):
    layer = PlainFilter(16)
    return early_stack(layer, NeuralNoise(sc)), layer, False

class FrozenBlur(nn.Module):
    """Fixed NON-zero-sum control: a Gaussian blur (the DoG's center alone, sums to 1) at the same
    strength. Tests whether the protection comes from smoothing or from the zero-sum surround."""
    def __init__(self, channels, alpha, k=5):
        super().__init__()
        g = torch.linspace(-(k//2), k//2, k); d = g[:, None]**2 + g[None, :]**2
        c = torch.exp(-d/(2*(k/6)**2)); c = c/c.sum()
        self.register_buffer("weight", c.repeat(channels, 1, 1, 1))
        self.register_buffer("alpha", torch.tensor([float(alpha)]))
        self.channels, self.pad = channels, k//2
    def forward(self, x):
        return x + self.alpha * F.conv2d(x, self.weight, padding=self.pad, groups=self.channels)

def plain_lowlr_noise(sc):
    layer = PlainFilter(16); layer.lr = 1e-4
    return early_stack(layer, NeuralNoise(sc)), layer, False

def zerosum_learned_noise(sc):
    layer = ZeroSumInhibition(16)
    return early_stack(layer, NeuralNoise(sc)), layer, True

def blur_noise(sc):
    return early_stack(FrozenBlur(16, 0.5), NeuralNoise(sc)), None, False

def control():
    return ControlCNN(), None, False

def early_frozen(alpha):
    m = ExperimentalCNN(use_inhib=False, use_attn=False)
    layer = FrozenInhibition(16, alpha)
    m.features = nn.Sequential(m.features[0], layer, *m.features[1:])
    return m, layer, False

def early_plain():
    m = ExperimentalCNN(use_inhib=False, use_attn=False)
    layer = PlainFilter(16)
    m.features = nn.Sequential(m.features[0], layer, *m.features[1:])
    return m, layer, False

def early(nonneg):
    # Right after the stem convolution (16 channels at 112x112), the closest stage to the retina.
    m = ExperimentalCNN(use_inhib=False, use_attn=False)
    inhib = ZeroSumInhibition(16)
    m.features = nn.Sequential(m.features[0], inhib, *m.features[1:])
    return m, inhib, nonneg
# Control, Early, a>=0 and plain filter come from iterations 4-5 (same seeds, subsets and corrupted images).
# Strengths: 0.1 matches what the learnable early layer chose (+0.04 to +0.17); 0.5 is a strong prior.
def lr_set(make, lr):
    m, layer, nn_ = make(); layer.lr = lr; return m, layer, nn_
# Matched learning rate (1e-3) for both learnable layers, both trained with internal noise 0.03.
configs = {"Learned zero-sum lr1e-3": lambda: lr_set(lambda: zerosum_learned_noise(0.03), 1e-3),
           "Plain lr1e-3": lambda: lr_set(lambda: plain_noise(0.03), 1e-3)}
def noise(s):
    return lambda x, g: norm(torch.clamp(x + torch.randn(x.shape, generator=g)*s, 0, 1))
def occlude(x, g):
    x = norm(x).clone()
    for i in range(len(x)):
        r, c = torch.randint(0, 224-50, (2,), generator=g).tolist(); x[i, :, r:r+50, c:c+50] = 0
    return x
conds = {"clean": lambda x, g: norm(x), "s=0.01": noise(0.01), "s=0.03": noise(0.03),
         "s=0.05": noise(0.05), "s=0.1": noise(0.1), "occluded": occlude}

SEEDS = list(range(3, 13))
res = {m: {c: [] for c in conds} for m in configs}
for seed in SEEDS:
    idx = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed))[:20000].tolist()
    for name, make in configs.items():
        torch.manual_seed(seed)
        m, inhib, nonneg = make(); m.to(dev)
        # New parameters (inhibition kernel, its strength, the attention/inhibition mix) get a higher
        # learning rate; at 1e-4, alpha could move at most ~0.03 in one epoch and never engage.
        new_ids = {id(p) for p in inhib.parameters()} if inhib is not None else set()
        new = [p for p in m.parameters() if id(p) in new_ids]
        rest = [p for p in m.parameters() if id(p) not in new_ids]
        opt = torch.optim.AdamW([{"params": rest, "lr": 1e-4}, {"params": new, "lr": getattr(inhib, "lr", 1e-2)}],
                                weight_decay=1e-5)
        m.train(); t0 = time.time()
        for x, y in DataLoader(Subset(train_full, idx), 64, shuffle=True,
                               generator=torch.Generator().manual_seed(seed)):
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            nn.functional.cross_entropy(m(x), y).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if nonneg:  # projected gradient: alpha can sit at 0 but never go negative
                with torch.no_grad(): inhib.alpha.clamp_(min=0)
        m.eval(); line = []
        for cname, f in conds.items():
            g = torch.Generator().manual_seed(1000 + seed)  # same corrupted images for every model
            c = n = 0
            with torch.no_grad():
                for x, y in DataLoader(test_px, 250):
                    c += (m(f(x, g).to(dev)).argmax(1).cpu() == y).sum().item(); n += len(y)
            res[name][cname].append(100*c/n); line.append(f"{cname} {100*c/n:.1f}")
        state = f"alpha={inhib.alpha.item():+.3f}" if inhib is not None else "-"
        print(f"seed {seed} {name:16s} ({time.time()-t0:.0f}s) " + " | ".join(line) + f"  [{state}]", flush=True)

json.dump(res, open("iter11_results.json", "w"), indent=1)
prev = json.load(open(sys.argv[2])); res = {"Control": prev["Control"], **res}
from scipy import stats
def compare(a_name, b_name):
    print(f"\n{b_name} minus {a_name} (paired by seed, same corrupted images)")
    print(f"{'condition':10s} {a_name[:13]:>13s} {b_name[:13]:>13s} {'diff':>7s} {'95% CI':>17s} {'p (paired t)':>13s} {'wins':>6s}")
    for c in conds:
        A, B = np.array(res[a_name][c]), np.array(res[b_name][c]); d = B - A
        lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        print(f"{c:10s} {A.mean():6.1f} ± {A.std(ddof=1):4.1f} {B.mean():6.1f} ± {B.std(ddof=1):4.1f} "
              f"{d.mean():+7.1f} [{lo:+6.1f}, {hi:+6.1f}] {stats.ttest_rel(B, A).pvalue:13.4f} {int((d > 0).sum()):>3d}/{len(d)}")
compare("Control", "Learned zero-sum lr1e-3")
compare("Control", "Plain lr1e-3")
compare("Plain lr1e-3", "Learned zero-sum lr1e-3")
