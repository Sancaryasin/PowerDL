import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from powerdl.highlevel import profile_torch  # <-- unified entrypoint

BS = 256
EPOCHS = 3

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
trainloader = DataLoader(
    trainset,
    batch_size=BS,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torchvision.models.resnet18(num_classes=10).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -------------------------
# PowerDL
# -------------------------
# NOTE: out_dir is set so raw artifacts (CSV/JSON) can also be exported by the profiler.
with profile_torch(out_dir="results/torch_cifar_profiler_raw", interval_s=0.02, verbose=1) as prof:
    prof.train(model, trainloader, optimizer, loss_fn, epochs=EPOCHS)
    prof.infer_tensor(model, batch_size=BS, n_samples=20000)

rep = prof.report()

# Figures (PNG)
rep.plot(all=True, out_dir="results/torch_cifar_figs", show=False, smooth=5, shade_phases=True)

# Reproducibility artifacts (CSV + summary JSON)
rep.export(
    "results/torch_cifar_run",
    include_csv=True,
    include_summary=True,
    include_figures=False,  # figures already produced above
)

print("Done.")
print("Profiler raw artifacts -> results/torch_cifar_profiler_raw")
print("Figures -> results/torch_cifar_figs")
print("CSV/Summary -> results/torch_cifar_run")
print("Available figure keys:", rep.list_figures())