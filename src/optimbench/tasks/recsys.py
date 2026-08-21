import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from optimbench.tasks import Task


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, mf_dim=16, mlp_dim=32, mlp_layers=(64, 32, 16)):
        super().__init__()
        self.mf_user = nn.Embedding(n_users, mf_dim)
        self.mf_item = nn.Embedding(n_items, mf_dim)
        self.mlp_user = nn.Embedding(n_users, mlp_dim)
        self.mlp_item = nn.Embedding(n_items, mlp_dim)
        layers = []
        in_dim = mlp_dim * 2
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(mf_dim + mlp_layers[-1], 1)

    def forward(self, user, item):
        mf = self.mf_user(user) * self.mf_item(item)
        mlp_in = torch.cat([self.mlp_user(user), self.mlp_item(item)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        return torch.sigmoid(self.out(torch.cat([mf, mlp_out], dim=-1))).squeeze(-1)


class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, users, items, labels):
        self.users = users
        self.items = items
        self.labels = labels

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


class MovieLensTask(Task):
    name = "movielens_neumf"
    higher_is_better = True
    max_steps_cap = 15000
    checkpoint_every = 150

    def __init__(self, ratings_path=None, batch_size=256, n_users=6040, n_items=3706):
        self.ratings_path = ratings_path
        self.batch_size = batch_size
        self.n_users = n_users
        self.n_items = n_items

    def build_data(self, seed):
        df = pd.read_csv(self.ratings_path, sep="::", engine="python", names=["user", "item", "rating", "ts"])
        df["user"] = df["user"].astype("category").cat.codes
        df["item"] = df["item"].astype("category").cat.codes
        self.n_users = df["user"].nunique()
        self.n_items = df["item"].nunique()
        df["label"] = (df["rating"] >= 4).astype("float32")
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = int(len(df) * 0.9)
        train_df, val_df = df.iloc[:n], df.iloc[n:]
        train_ds = RatingDataset(
            torch.tensor(train_df["user"].values, dtype=torch.long),
            torch.tensor(train_df["item"].values, dtype=torch.long),
            torch.tensor(train_df["label"].values, dtype=torch.float32),
        )
        val_ds = RatingDataset(
            torch.tensor(val_df["user"].values, dtype=torch.long),
            torch.tensor(val_df["item"].values, dtype=torch.long),
            torch.tensor(val_df["label"].values, dtype=torch.float32),
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False)
        return train_loader, val_loader

    def build_model(self, seed):
        torch.manual_seed(seed)
        return NeuMF(self.n_users, self.n_items)

    def loss_and_metric(self, model, batch):
        user, item, label = batch
        pred = model(user, item)
        return F.binary_cross_entropy(pred, label)

    def evaluate(self, model, val_loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for user, item, label in val_loader:
                pred = model(user, item)
                correct += ((pred >= 0.5).float() == label).sum().item()
                total += label.size(0)
        model.train()
        return correct / total
