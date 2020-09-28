import torch

class Model(torch.nn.Module):
    def __init__(self, classnum, embeddims, device):
        super(Model, self).__init__()
        self.idxes = [i for i, d in enumerate(embeddims) if d > 0]
        self.embeddim = sum(embeddims)
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(n, d) for n, d in zip(classnum, embeddims)])
        self.to(device)

    def __call__(self, X):
        if self.embeddim > 0:
            return torch.cat([self.embeddings[i](X[i:i+1])[0] for i in self.idxes])
        else:
            return torch.empty((0,), dtype=torch.float32).to(X.device)
