import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from powerdl.highlevel_torch import profile_torch

BS = 256
EPOCHS = 3

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torchvision.models.resnet18(num_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

with profile_torch(out_dir=None, interval_s=0.02, verbose=1) as prof:
    prof.train(model, trainloader, optimizer, loss_fn, epochs=EPOCHS)
    prof.infer_tensor(model, batch_size=BS, n_samples=20000)

rep = prof.report()

# Produce a rich set of figures (skip gracefully when a metric isn't available).
rep.plot(all=True, out_dir="results/torch_cifar_figs", show=False, smooth=5, shade_phases=True)

# Optional: persist raw artifacts for reproducibility
# rep.export("runs/torch_cifar", include_csv=True, include_summary=True, include_figures=True)

print("Done. Figures -> results/torch_cifar_figs")
print("Available figure keys:", rep.list_figures())
