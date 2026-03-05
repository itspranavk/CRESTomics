import torch
import numpy as np
from scipy.stats import norm
from itertools import product
from tqdm import tqdm

from .loss import CoherenceLoss

class CoherenceAdditiveModel:
    def __init__(self, max_iter=200, tol=1e-3, lam=0.005, sigma=0.5, gamma_kernel=1.5, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_iter = max_iter
        self.tol = tol
        self.lam = lam
        self.sigma = sigma
        self.loss_fn = CoherenceLoss(sigma, device=self.device)
        self.w = None
        self.alpha = None
        self.gamma = None
        self.gamma_kernel = gamma_kernel
        self.K_groups = None
        self.n = None
        self.d = None
    
    def rbf_kernel(self, X, Y):
        XX = (X ** 2).sum(1).view(-1, 1)
        YY = (Y ** 2).sum(1).view(1, -1)
        dist = XX + YY - 2 * X @ Y.T
        return torch.exp(-self.gamma_kernel * dist)
        
    def get_K_groups(self, X):     
        K_groups = []
        norms = []
        for grp in self.groups:
            K = self.rbf_kernel(X[:, grp], X[:, grp])
            norm = K.norm().to(self.device)
            K_groups.append(K / norm)
            norms.append(norm)
        
        self.K_norms = norms
        return K_groups

    def f(self, group_idx=None):
        f = torch.zeros(self.n, device=self.device)
        if group_idx is None:
            for K, a in zip(self.K_groups, self.alpha):
                f += K @ a
        else:
            f = self.K_groups[group_idx] @ self.alpha[group_idx]
        return f

    def grad_L(self):
        f = self.f()
        grad_f = self.loss_fn.grad(self.y, f, sample_weight=self.sample_weight)
        grad_groups = [K.T @ grad_f / self.n for K in self.K_groups]
        return grad_groups

    def step(self):
        U = [-g for g in self.grad_L()]
        
        for j in range(self.d):
            gamma_j = self.gamma[j]
            aj = self.alpha[j]
            Sj = U[j] + gamma_j * aj
            normS = Sj.norm(2)
            
            if normS > 1e-10:
                shrink = max(0.0, 1 - self.lam * self.w[j] / normS)
                self.alpha[j] = (Sj / gamma_j) * shrink
            else:
                self.alpha[j] = torch.zeros_like(aj, device=self.device)

    def fit(self, X, y, groups=None, sample_weight=None, progress=True, verbose=False):
        # Convert inputs to tensor
        self.X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        self.y = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        self.y = torch.where(self.y == 0, -1.0, 1.0)
        
        self.n = self.y.numel()
        self.groups = groups or [[j] for j in range(self.X.shape[1])]
        
        if sample_weight is not None:
            self.sample_weight = torch.as_tensor(sample_weight, dtype=torch.float32, device=self.device)
        else:
            self.sample_weight = None

        # Compute kernel matrices for each group
        self.K_groups = self.get_K_groups(self.X)
        self.d = len(self.K_groups)
        self.w = torch.ones(self.d, device=self.device)

        # Init alphas
        self.alpha = [torch.zeros(self.n, device=self.device) for _ in range(self.d)]

        # Compute curvatures for each group
        self.gamma = []
        for K in self.K_groups:
            xi_j = torch.linalg.norm(K, 2).real
            self.gamma.append((1 + 1e-6) * xi_j)

        # Groupwise-majorization descent
        iter_fn = range(self.max_iter)
        if progress:
            iter_fn = tqdm(iter_fn, desc='Iter')
            
        for it in iter_fn:
            old_alpha = [a.clone() for a in self.alpha]
            
            self.step()
            
            diff = torch.stack([(a - a_old).norm() for a, a_old in zip(self.alpha, old_alpha)]).sum()
            
            if verbose and it % 1000 == 0:
                fval = self.f()
                loss = self.loss_fn(self.y, fval).item()
                print(f"Iter {it:03d} | Loss = {loss:.4f} | diff = {diff:.3e}")
                
            if diff < self.tol:
                if verbose:
                    print(f'Converged at iter {it}')
                break
            
        return self
    
    def gridsearch(
        self, 
        X, 
        y,
        groups = None,
        sample_weight = None,
        gammas = np.linspace(0.1, 2.0, 5),
        sigmas = np.linspace(0.1, 1.0, 5),
        lams = np.logspace(-3, 3, 11),
        progress = True,
        verbose = False,
    ):
        best_acc = -float('inf')
        best_params = None
        
        params = list(product(gammas, sigmas, lams))

        iter_fn = params
        if progress:
            iter_fn = tqdm(iter_fn, desc='Grid Search')
   
        for gamma, sigma, lam in iter_fn:
            # Create temporary model
            model = CoherenceAdditiveModel(
                max_iter=self.max_iter,
                lam=lam,
                sigma=sigma,
                gamma_kernel=gamma,
                device=self.device
            )

            # Fit with current hyperparameters
            model.fit(X, y, groups, sample_weight, progress=False, verbose=False)

            # Evaluate performance (accuracy on training set)
            y_pred = model.predict(X)
            acc = (y_pred == y).mean().item()

            # Update best
            if acc > best_acc:
                best_acc = acc
                best_params = [gamma, sigma, lam]

        # Set optimal hyperparameters
        self.gamma_kernel = best_params[0]
        self.sigma = best_params[1]
        self.lam = best_params[2]
        
        # Reset all others
        self.loss_fn = CoherenceLoss(self.sigma, device=self.device)
        self.w = None
        self.alpha = None
        self.gamma = None
        self.K_groups = None
        self.n = None
        self.d = None
        
        # Fit with optimal hyperparameters
        self.fit(X, y, groups, sample_weight, progress, verbose)
        
        return self

    def predict_scores(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        
        if self.groups is None:
            self.groups = [[j] for j in range(X.shape[1])]
        
        f = torch.zeros(X.shape[0], device=self.device)
        
        for j, grp in enumerate(self.groups):
            Kt = self.rbf_kernel(X[:, grp], self.X[:, grp])
            Kt = Kt / self.K_norms[j]
            f += Kt @ self.alpha[j]
            
        return f.cpu()
    
    def predict_proba(self, X):
        scores = self.predict_scores(X)
        return torch.sigmoid(scores / self.sigma).cpu().numpy()

    def predict(self, X):
        scores = self.predict_scores(X)
        pred = torch.where(scores >= 0, 1, 0)
        return pred.cpu().numpy()
    
    def get_importance(self, group_idx):
        return torch.norm(self.alpha[group_idx]).item()
    
    def partial_dependence(self, group_idx, ci=0.95):
        # Group weight for jitter
        wj = 1.0 
        
        if isinstance(group_idx, int):
            cols = [group_idx]
            alpha_j = self.alpha[group_idx]
        else:
            cols = group_idx
            alpha_j = sum(self.alpha[j] for j in cols)

        # Generate grid over all selected features
        Xg = self.X[:, cols]
        grid = torch.linspace(Xg.min(), Xg.max(), 100, device=self.device).view(-1, len(cols))
        alpha_j = sum(self.alpha[j] for j in cols)

        # Compute kernel between grid and training points
        Kg = self.rbf_kernel(grid, Xg)

        # Partial dependence mean
        pdep = Kg @ alpha_j
        pdep -= pdep.mean()

        # Compute Hessian for all selected features
        f_train = self.predict_scores(self.X).to(device=self.device)
        W_diag = self.loss_fn.second_derivative(self.y, f_train)
        W = torch.diag(W_diag)

        # Sum Hessians of each feature in the group
        Hj = torch.zeros((self.n, self.n), device=self.device)
        for j in cols:
            Kj = self.rbf_kernel(self.X[:, [j]], self.X[:, [j]])
            Hj += Kj.T @ W @ Kj + self.lam * wj * torch.eye(self.n, device=self.device)

        # Invert Hessian (with jitter)
        Hj_inv = torch.linalg.inv(Hj + 1e-6 * torch.eye(self.n, device=self.device))

        # Variance per grid point
        var_pd = torch.sum((Kg @ Hj_inv) * Kg, dim=1)
        z = norm.ppf(1 - (1 - ci) / 2)
        ci = z * torch.sqrt(var_pd)
        ci = torch.stack([pdep - ci, pdep + ci], dim=1)

        # Detach and convert to np
        grid = grid.squeeze().cpu().numpy()
        pdep = pdep.detach().cpu().numpy()
        ci = ci.detach().cpu().numpy()

        return grid, pdep, ci