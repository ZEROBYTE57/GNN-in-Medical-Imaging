import sys, torch, torch_geometric, sksurv, lifelines, pandas
from torch_geometric.nn import GCNConv
print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| pyg", torch_geometric.__version__, "| sksurv", sksurv.__version__)
x = torch.randn(4, 3); ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
print("GCN output shape:", GCNConv(3, 2)(x, ei).shape)  # expect torch.Size([4, 2])