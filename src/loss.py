import torch

class CoherenceLoss:
    def __init__(self, sigma=0.5, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sigma = torch.tensor(float(sigma), device=self.device)
        self.den = torch.log(1 + torch.exp(1 / self.sigma))
        
    def __call__(self, y, f, sample_weight=None):
        loss = torch.log(1 + torch.exp((1 - y * f) / self.sigma)) / self.den
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()
    
    def grad(self, y, f, sample_weight=None):
        num = torch.exp((1 - y * f) / self.sigma)
        den = (self.sigma * (1 + num) * self.den)
        grad = -y * num / den
        if sample_weight is not None:
            grad = grad * sample_weight
        return grad
    
    def second_derivative(self, y, f):
        num = torch.exp((1 - y * f) / self.sigma)
        den = (1 + num)**2 * (self.sigma**2 * self.den)
        return num / den