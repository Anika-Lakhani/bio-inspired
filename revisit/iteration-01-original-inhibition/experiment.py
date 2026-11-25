# Paper's models (classes loaded verbatim from internal-code.ipynb), with the two bugs fixed in the harness:
#  - noise added in pixel space before normalization (no clamp of normalized values)
#  - ablation flags passed explicitly so each variant is really what its name says
import json, sys, time, torch, torch.nn as nn, torchvision, torchvision.transforms as T, numpy as np
from torch.utils.data import DataLoader, Subset

nb = json.load(open(sys.argv[1]))
ns = {"__name__": "nb"}
exec("".join(nb["cells"][0]["source"]), ns)
ControlCNN, ExperimentalCNN = ns["ControlCNN"], ns["ExperimentalCNN"]

dev = torch.device("mps")
norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
base = T.Compose([T.Resize((224,224)), T.ToTensor()])
train_full = torchvision.datasets.CIFAR10("./data", True, transform=T.Compose([base, norm]))
test_px = Subset(torchvision.datasets.CIFAR10("./data", False, transform=base), range(5000))

configs = {
    "Control": lambda: ControlCNN(),
    "Attention Only": lambda: ExperimentalCNN(use_inhib=False, use_attn=True),
    "Inhibition Only": lambda: ExperimentalCNN(use_inhib=True, use_attn=False),
    "Combined": lambda: ExperimentalCNN(use_inhib=True, use_attn=True),
}
def noise(s):
    return lambda x, g: norm(torch.clamp(x + torch.randn(x.shape, generator=g)*s, 0, 1))
def occlude(x, g):
    x = norm(x).clone()
    for i in range(len(x)):
        r, c = torch.randint(0, 224-50, (2,), generator=g).tolist(); x[i, :, r:r+50, c:c+50] = 0
    return x
conds = {"clean": lambda x, g: norm(x), "s=0.01": noise(0.01), "s=0.03": noise(0.03),
         "s=0.05": noise(0.05), "s=0.1": noise(0.1), "occluded": occlude}

SEEDS = [0, 1, 2]
res = {m: {c: [] for c in conds} for m in configs}
for seed in SEEDS:
    idx = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed))[:20000].tolist()
    for name, make in configs.items():
        torch.manual_seed(seed)
        m = make().to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)
        m.train(); t0 = time.time()
        for x, y in DataLoader(Subset(train_full, idx), 64, shuffle=True,
                               generator=torch.Generator().manual_seed(seed)):
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            nn.functional.cross_entropy(m(x), y).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval(); line = []
        for cname, f in conds.items():
            g = torch.Generator().manual_seed(1000 + seed)  # same corrupted images for every model
            c = n = 0
            with torch.no_grad():
                for x, y in DataLoader(test_px, 250):
                    c += (m(f(x, g).to(dev)).argmax(1).cpu() == y).sum().item(); n += len(y)
            res[name][cname].append(100*c/n); line.append(f"{cname} {100*c/n:.1f}")
        print(f"seed {seed} {name:16s} ({time.time()-t0:.0f}s) " + " | ".join(line), flush=True)

json.dump(res, open("multiseed_results.json", "w"), indent=1)
print("\nmean ± std over seeds (accuracy %), and paired diff vs Control")
print(f"{'':16s}" + "".join(f"{c:>18s}" for c in conds))
for name in configs:
    print(f"{name:16s}" + "".join(f"{np.mean(v):9.1f} ± {np.std(v, ddof=1):4.1f}   " for v in res[name].values()))
for name in list(configs)[1:]:
    print(f"{name+' - Ctrl':16s}" + "".join(
        f"{np.mean(d):+9.1f} ± {np.std(d, ddof=1):4.1f}   "
        for d in (np.array(res[name][c]) - np.array(res['Control'][c]) for c in conds)))
