import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from optimbench.tasks import Task


class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=5, num_classes=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dim = in_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            dim = hidden_dim
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        x = global_add_pool(x, batch)
        return self.head(x)


class MolHivTask(Task):
    name = "molhiv_gin"
    higher_is_better = True
    max_steps_cap = 15000
    checkpoint_every = 150

    def __init__(self, data_root, batch_size=64):
        self.data_root = data_root
        self.batch_size = batch_size

    def build_data(self, seed):
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=self.data_root)
        split = dataset.get_idx_split()
        self.evaluator = Evaluator(name="ogbg-molhiv")
        self.in_dim = dataset.num_features
        train_loader = DataLoader(dataset[split["train"]], batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[split["valid"]], batch_size=256, shuffle=False)
        return train_loader, val_loader

    def build_model(self, seed):
        torch.manual_seed(seed)
        return GIN(self.in_dim)

    def loss_and_metric(self, model, batch):
        out = model(batch.x.float(), batch.edge_index, batch.batch)
        return F.binary_cross_entropy_with_logits(out, batch.y.float())

    def evaluate(self, model, val_loader):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x.float(), batch.edge_index, batch.batch)
                y_true.append(batch.y)
                y_pred.append(out)
        model.train()
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        result = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred})
        return result["rocauc"]
