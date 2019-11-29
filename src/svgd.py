import numpy as np
import torch
import math


class SVGD:
  def __init__(self, distribution, kernel, optimizer):
    self.P = distribution
    self.K = kernel
    self.optim = optimizer

  def phi(self, X, *data):
    X = X.detach().requires_grad_(True)

    log_prob = self.P.log_prob(X, *data)
    score_func = torch.autograd.grad(log_prob.sum(), X)[0]

    K_XX = self.K(X, X.detach())
    grad_K = - torch.autograd.grad(K_XX.sum(), X)[0]

    phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

    return phi

  def step(self, particles, *data):
    self.optim.zero_grad()
    particles.grad = -self.phi(particles, *data)
    self.optim.step()



class RBF_Kernel(torch.nn.Module):
    r"""
      RBF kernel

      :math:`K(x, y) = exp(||x-v||^2 / (2h))

      """

    def __init__(self, bandwidth=None):
        super().__init__()
        self.bandwidth = bandwidth

    def _bandwidth(self, norm_sq):
        # Apply the median heuristic (PyTorch does not give true median)
        if self.bandwidth is None:
            np_dnorm2 = norm_sq.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(np_dnorm2.shape[0] + 1))
            return np.sqrt(h).item()
        else:
            return self.bandwidth

    def forward(self, X, Y):
        dnorm2 = norm_sq(X, Y)
        bandwidth = self._bandwidth(dnorm2)
        gamma = 1.0 / (1e-8 + 2 * bandwidth ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY



class IMQSteinKernel(torch.nn.Module):
    r"""
    IMQ (inverse multi-quadratic) kernel

    :math:`K(x, y) = (\alpha + ||x-y||^2/h)^{\beta}`

    """

    def __init__(self, alpha=0.5, beta=-0.5, bandwidth=None):
        super(IMQSteinKernel, self).__init__()
        assert alpha > 0.0, "alpha must be positive."
        assert beta < 0.0, "beta must be negative."
        self.alpha = alpha
        self.beta = beta
        self.bandwidth = bandwidth

    def _bandwidth(self, norm_sq):
        """
        Compute the bandwidth along each dimension using the median pairwise squared distance between particles.
        """
        if self.bandwidth is None:
            num_particles = norm_sq.size(0)
            index = torch.arange(num_particles)
            norm_sq = norm_sq[index > index.unsqueeze(-1), ...]
            median = norm_sq.median(dim=0)[0]
            assert median.shape == norm_sq.shape[-1:]
            return median / math.log(num_particles + 1)
        else:
            return self.bandwidth

    def forward(self, X, Y):
        norm_sq = (X.unsqueeze(0) - Y.unsqueeze(1))**2  # N N D
        assert norm_sq.dim() == 3
        bandwidth = self._bandwidth(norm_sq)  # D
        base_term = self.alpha + torch.sum(norm_sq / bandwidth, dim=-1)
        log_kernel = self.beta * torch.log(base_term)  # N N D
        return log_kernel.exp()

""" Helpers """

def norm_sq(X, Y):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())
    return -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)
