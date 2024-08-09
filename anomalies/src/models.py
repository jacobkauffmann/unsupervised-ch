import torch
import torch.nn as nn
import torch as tr
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.cluster import KMeans
from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
import numpy as np
import patchcore
import patchcore.patchcore
import patchcore.backbones
# from patchcore.common import SklearnNN, FaissNN
from patchcore.common import FaissNN
import warnings

from src.utils import interval_cutting

def lme(D, s, dim=-1):
    # return (1/s)*(tr.logsumexp(s*D, dim=dim) - np.log(D.shape[dim]))
    return (1/s)*(torch.logsumexp(s*D, dim=dim) - torch.log(torch.tensor(D.shape[dim], dtype=torch.float32)))


class MahalanobisKDE(nn.Module):
    def __init__(self, svs=None, gamma=None, layer=None, **kwargs):
        super().__init__()
        if svs is not None:
            self.svs = svs.clone().detach()
        if gamma is not None:
            self.gamma = gamma
        else:
            self.gamma = torch.tensor(1.)
        if layer is not None:
            self.layer = layer
        else:
            self.layer = nn.Identity()

        self.p = 2

    def fit(self, X, fit_gamma=True, verbose=False, **kwargs):
        self.svs = X.clone()
        if fit_gamma:
            # N, D = self.svs.flatten(1).shape
            # gamma = 1.
            # dist = torch.cdist(
            #     self.layer(self.svs).flatten(1),
            #     self.layer(self.svs).flatten(1))**2
            # dist = dist[~torch.eye(N).bool()].reshape(N, N - 1)
            # progress = float('inf')
            # ll = self.logpdf(dist, gamma).item()
            # ctr = 0
            # while progress > 1e-6:
            #     llold = ll
            #     w = torch.nn.functional.softmax(-gamma * dist, dim=1)
            #     gamma_inv = 2 / (N * D) * (w * dist).sum()
            #     gamma = 1 / gamma_inv
            #     ll = self.logpdf(dist, gamma).item()
            #     progress = abs(llold - ll)
            #     ctr += 1
            # if verbose:
            #     print("Bandwidth was set to %f after %d iterations" % (gamma, ctr))
            # self.gamma = gamma
            def perplexity(D, gamma):
                N = D.shape[0]
                total_perplexity = 0.0

                P = tr.exp(-gamma*D)
                P.fill_diagonal_(0)

                P_sum = P.sum(dim=1, keepdim=True)
                P_sum[P_sum == 0] = 1
                P = P / P_sum

                H_P = -tr.sum(P * tr.log(P + 1e-10), dim=1)
                perplexity = H_P.exp().mean()

                return perplexity.item()

            D = tr.cdist(X.flatten(1), X.flatten(1), p=self.p)**self.p
            avg_perp = lambda gamma: perplexity(D, gamma) - 0.25*len(X)

            self.gamma = interval_cutting(1e-9, 1.0, avg_perp, verbose=True)
        return self

    def logpdf(self, dist, gamma):
        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma)
        _, D = self.svs.flatten(1).shape
        logp = torch.logsumexp(-gamma * dist,
                               dim=1) + (D / 2) * torch.log(gamma)
        return logp.sum()

    def h(self, x):
        x = torch.cdist(self.layer(x).flatten(1), self.layer(self.svs).flatten(1), p=self.p)**self.p
        return x

    def forward(self, X):
        # return tr.stack([
        #     lme(self.h(x.unsqueeze(0)), -self.gamma).squeeze()
        #     for x in X
        # ])
        return lme(self.h(X), -self.gamma) # this is buggy on MTS!!!

    def explain(self, x, eps=1e-6):
        self.svs.requires_grad_(True)
        self.svs.grad = None
        delta = x - self.svs

        o = self.forward(x)
        factor = o / ((delta.flatten(1)**2).sum(1) + eps)
        factor = factor.view(-1, *([1]*(delta.dim()-1)))
        o.backward()
        return -.5 * (delta * self.svs.grad * factor).sum(0)

class D2NeighborsL1(MahalanobisKDE):
    def __init__(self, svs=None, gamma=None, layer=None, **kwargs):
        super().__init__(svs=svs, gamma=gamma, layer=layer, **kwargs)
        self.p = 1

    # def explain(self, x, eps=1e-6):
    #     self.svs.requires_grad_(True)
    #     self.svs.grad = None
    #     delta = x - self.svs

    #     o = self.forward(x)
    #     factor = o / (delta.flatten(1).abs().sum(1) + eps)
    #     factor = factor.view(-1, *([1]*(delta.dim()-1)))
    #     o.backward()
    #     return -(delta * self.svs.grad * factor).sum(0)


class D2NeighborsL4(MahalanobisKDE):
    def __init__(self, svs=None, gamma=None, layer=None, **kwargs):
        super().__init__(svs=svs, gamma=gamma, layer=layer, **kwargs)
        self.p = 4

    # def explain(self, x, eps=1e-6):
    #     self.svs.requires_grad_(True)
    #     self.svs.grad = None
    #     delta = x - self.svs

    #     o = self.forward(x)
    #     factor = o / ((delta.flatten(1)**4).sum(1) + eps)
    #     factor = factor.view(-1, *([1]*(delta.dim()-1)))
    #     o.backward()
    #     return -(1/4) * (delta * self.svs.grad * factor).sum(0)


# class D2Neighbors2(MahalanobisKDE):
#     def explain(self, x):
#         self.svs.requires_grad_(True)
#         self.svs.grad = None
#         delta = x - self.svs

#         o = self.forward(x)
#         o.backward()
#         return -.5 * delta.mean(0) * self.svs.grad.sum(0)

class MLP(nn.Module):
    def __init__(self, layer=lambda x: x, hidden_layer_sizes=(100,), max_iter=400, random_state=42):
        super().__init__()
        self.max_iter = max_iter
        self.layer = layer
        # self.clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state, activation='relu', alpha=0.01)
        self.conv1 = nn.Conv2d(3, 100, 50)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(100, 1, 1)

    def fit(self, X, y, batch_size=32, **kwargs):
        # do this on mps
        device = torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.to(device)
        X = X.clone().to(device)
        y = y.clone().to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-2)
        criterion = torch.nn.BCEWithLogitsLoss()
        y = y.view(-1, 1, 1, 1).expand(-1, 1, 15, 15).float()
        # for _ in range(self.max_iter):
        #     optimizer.zero_grad()
        #     yhat = self(X)
        #     loss = criterion(yhat, y)
        #     loss.backward()
        #     optim.step()
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(self.max_iter):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                yhat = self(batch_X)
                loss = criterion(yhat, batch_y)
                print(loss.item())
                loss.backward()
                optimizer.step()
        self.to('cpu')
        return self

    def forward(self, X):
        X = self.layer(X)
        X = self.conv1(X)
        X = self.relu1(X)
        X = self.conv2(X)
        return X

    def explain(self, x):
        x.requires_grad_(True)
        x.grad = None
        y = self.forward(x).sum()
        y.backward()
        return x*x.grad

# class Supervised(nn.Module):
#     def __init__(self, gamma=1, layer=None):
#         super().__init__()
#         self.gamma = gamma
#         if layer is not None:
#             self.layer = layer
#         else:
#             self.layer = nn.Identity()

#     def fit(self, X, y, **kwargs):
#         self.svs = X
#         self.y = y

#     def forward(self, X):
#         if X.dim() == 1:
#             X = X.unsqueeze(0)
#         N = len(X)
#         C = tr.unique(self.y)
#         K = len(C)
#         P = tr.zeros(N, K)
#         distance = tr.cdist(self.layer(X), self.layer(self.svs))**2
#         for cidx, c in enumerate(C):
#             indices = self.y == c
#             P[:, cidx] = lme(distance[:, indices], -self.gamma)
#         P = -(P[:,:,None] - P[:,None,:])
#         P = P[:,~tr.eye(K).bool()].reshape(N, K, K-1)
#         # P = lme(P, -self.gamma)
#         P = P.min(-1).values
#         return P

#     def explain(self, x, c):
#         delta = x - self.svs
#         self.svs.requires_grad_(True)
#         self.svs.grad = None
#         P = self.forward(x)
#         # C = (P[:, c] / self.layer(delta).norm(p=2, dim=-1)**2).clamp(max=1)
#         # C = C.view(-1, *([1]*(delta.dim()-1)))
#         # P[:, c].backward()
#         # return -.5 * (delta * self.svs.grad * C).sum(0)
#         P[:,c].backward()
#         return -.5 * (delta * self.svs.grad).sum(0)

    # def __init__(self, kernel='linear', C=1e-5, gamma=None):
    #     self.gamma = gamma
    #     self.clf = SVC(kernel=kernel, gamma=gamma, C=C, class_weight='balanced')
    #     # self.clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    #     # self.clf = KNeighborsClassifier(n_neighbors=50)
    #     # self.clf = NearestCentroid()

    # def fit(self, X, y):
    #     X = X.flatten(1).numpy()
    #     self.clf.fit(X, y)

    # def predict(self, X):
    #     X = X.flatten(1).numpy()
    #     return self.clf.decision_function(X)
    #     # # return self.clf.predict_log_proba(X)[:,1]
    #     # # return self.clf.predict_proba(X)[:,1]
    #     # C = self.clf.centroids_
    #     # D0 = np.linalg.norm(X - C[0], axis=1)**2
    #     # D1 = np.linalg.norm(X - C[1], axis=1)**2
    #     # return D1 - D0

    # def train_test_split(self, X_train, X_test, y_test):
    #     tr.manual_seed(42)
    #     permutation = tr.randperm(len(X_test))
    #     positive_indices = permutation[y_test[permutation]]
    #     n = len(positive_indices) // 2

    #     y_train = tr.cat([tr.zeros(len(X_train)), tr.ones(n)])
    #     X_train = tr.cat([X_train, X_test[positive_indices[:n]]])

    #     test_indices = [i for i in range(len(X_test)) if i not in positive_indices[:n].tolist()]
    #     X_test = X_test[test_indices]
    #     y_test = y_test[test_indices]
    #     y_test = y_test.type_as(y_train)

    #     return X_train, y_train, X_test, y_test

# class KernelClustering():
#     def __init__(self, gamma=1):
#         self.gamma = gamma

#     def fit_svs(self, X, y, svs_per_class=10):
#         yy = tr.unique(y)
#         svs = tr.cat([
#             tr.from_numpy(KMeans(n_clusters=svs_per_class, random_state=42).fit(X[y == c]).cluster_centers_)
#             for c in yy
#         ]).float()
#         ysvs = tr.cat([tr.full((svs_per_class,), c) for c in yy])
#         return svs, ysvs

#     def train_test_split(self, X_train, X_test, y_test):
#         tr.manual_seed(42)
#         permutation = tr.randperm(len(X_test))
#         positive_indices = permutation[y_test[permutation]]
#         n = len(positive_indices) // 2

#         y_train = tr.cat([tr.zeros(len(X_train)), tr.ones(n)])
#         X_train = tr.cat([X_train, X_test[positive_indices[:n]]])

#         test_indices = [i for i in range(len(X_test)) if i not in positive_indices[:n].tolist()]
#         X_test = X_test[test_indices]
#         y_test = y_test[test_indices]
#         y_test = y_test.type_as(y_train)

#         return X_train, y_train, X_test, y_test

#     def lme(self, x, s, axis=-1):
#         return (1/s)*(tr.logsumexp(s*x, axis=axis) - np.log(x.shape[axis]))

#     def h(self, x, Xc, Xk):
#         Dc = tr.cdist(x, Xc)**2
#         Dk = tr.cdist(x, Xk)**2
#         return Dk[:,:,None] - Dc[:,None,:]

#     def fc(self, x, c, X, Y):
#         X = X.flatten(1)
#         x = x.flatten(1)
#         YY = tr.unique(Y)
#         knc = [k for k in YY if k!=c]
#         hk = tr.empty(len(YY)-1)
#         for k, yk in enumerate(knc):
#             hijk = self.h(x, X[Y==c], X[Y==yk])
#             hjk  = self.lme(hijk, self.gamma)
#             hk[k] = self.lme(hjk, -self.gamma)
#         f = hk.min(axis=-1).values
#         return f

#     def explain(self, x, c, fc, X, Y):
#         X = X.flatten(1)
#         x = x.flatten(1)
#         YY = tr.unique(Y)
#         knc = [k for k in YY if k!=c]
#         Dc = tr.cdist(x, X[Y==c])**2
#         Rc = tr.softmax(-self.gamma*Dc, axis=-1).flatten()
#         R = tr.zeros_like(x)
#         for k, yk in enumerate(knc):
#             Dk = tr.cdist(x, X[Y==yk])**2
#             Rk = tr.softmax(-self.gamma*Dk, axis=-1).flatten()
#             Rijk = Rc[:,None]*Rk[None,:]*fc
#             hijk = Dk[0,:,None] - Dc[0,None,:]
#             hjk  = self.lme(hijk, self.gamma)
#             hk   = self.lme(hjk, -self.gamma)
#             Rijk = Rijk * (hijk.T / hk).abs()
#             for i, idx in enumerate((Y==c).nonzero().squeeze()):
#                 for j, jdx in enumerate((Y==yk).nonzero().squeeze()):
#                     R += (2*(X[idx]-X[jdx])*x - X[idx]**2 + X[jdx]**2)*Rijk[i,j]
#         return R

device = torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

class PatchCore():
    def __init__(self, layer=lambda x: x, backbone="wideresnet50", imsize=(64,64), device=device, **kwargs):
            self.device = device
            self.model = patchcore.patchcore.PatchCore(device=device)
            self.layer = layer
            self.backbone = patchcore.backbones.load(backbone)

            self.model.load(
                backbone=self.backbone,
                layers_to_extract_from=['layer2'],
                device=device,
                input_shape=(3, *imsize),
                pretrain_embed_dimension=2048,
                target_embed_dimension=2048,
                # nn_method=SklearnNN(4)
                # nn_method=FaissNN(4)
            )

    def fit(self, X, *args, **kwargs):
        X = self.layer(X)
        # turn into dataloader
        dataloader = torch.utils.data.DataLoader(X, batch_size=len(X), shuffle=True)
        self.model.fit(dataloader)

    def forward(self, x):
        self(x)

    def __call__(self, x):
        x = self.layer(x)
        y, _ = self.model.predict(x)
        y = torch.tensor(y)
        return y

models = {
    'D2Neighbors': MahalanobisKDE,
    # 'D2Neighbors2': D2Neighbors2,
    'D2NeighborsL1': D2NeighborsL1,
    'D2NeighborsL4': D2NeighborsL4,
    # 'Supervised': Supervised,
    'PatchCore': PatchCore,
    'MLP': MLP
}
