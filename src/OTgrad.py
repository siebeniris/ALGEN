import torch
import numpy as np
import ot
from typing import Tuple, Optional, Dict
from utils import pairwise_cosine


cos = torch.nn.CosineSimilarity(dim=1)

class OptimalTransportGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sim: torch.Tensor, reg: float = 0.1, reg_m: float = 0.01,
                strategy: str = "ub_sinkhorn") -> torch.Tensor:
        """
        Forward pass for optimal transport with gradient support.

        Args:
            sim: Similarity matrix (n x m)
            reg: Entropic regularization parameter
            reg_m: Marginal regularization parameter
            strategy: OT strategy ("ub_sinkhorn", "gw", or "emd")

        Returns:
            Transport plan matrix
        """
        ctx.save_for_backward(sim)
        ctx.reg = reg
        ctx.reg_m = reg_m
        ctx.strategy = strategy

        # Convert to numpy for POT library
        M = -sim.detach().cpu().numpy()
        n, m = M.shape
        a = np.ones(n) / n
        b = np.ones(m) / m

        if strategy == "ub_sinkhorn":
            weight = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg, reg_m)
        elif strategy == "emd":
            weight = ot.emd(a, b, M)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        return torch.FloatTensor(weight).to(sim.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """
        Backward pass computing gradients.

        Args:
            grad_output: Gradient from subsequent layer

        Returns:
            Tuple of gradients for inputs and None for other parameters
        """
        sim, = ctx.saved_tensors

        # Approximate gradient based on strategy
        if ctx.strategy == "ub_sinkhorn":
            # Entropic gradient approximation
            grad_sim = -grad_output * torch.exp(-sim / ctx.reg)
        else:
            # Linear approximation for EMD
            grad_sim = -grad_output

        return grad_sim, None, None, None


def optimal_transport_weight(sim: torch.Tensor,
                             device: torch.device,
                             C1: Optional[torch.Tensor] = None,
                             C2: Optional[torch.Tensor] = None,
                             reg: float = 0.1,
                             reg_m: float = 0.01,
                             ot_strategy: str = "ub_sinkhorn",
                             requires_grad: bool = False) -> torch.Tensor:
    """
    Compute optimal transport weights with optional gradient computation.

    Args:
        sim: Similarity matrix
        device: PyTorch device
        C1: First cost matrix for GW (optional)
        C2: Second cost matrix for GW (optional)
        reg: Entropic regularization parameter
        reg_m: Marginal regularization parameter
        ot_strategy: OT strategy to use
        requires_grad: Whether to compute gradients

    Returns:
        Transport plan matrix
    """
    if requires_grad and ot_strategy != "gw":
        weight = OptimalTransportGrad.apply(sim, reg, reg_m, ot_strategy)
    else:
        with torch.no_grad():
            M = -sim.cpu().numpy()
            n, m = M.shape
            a = np.ones(n) / n
            b = np.ones(m) / m

            if ot_strategy == "ub_sinkhorn":
                weight = torch.FloatTensor(
                    ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg, reg_m))
            elif ot_strategy == "gw":
                if C1 is None or C2 is None:
                    raise ValueError("C1 and C2 required for GW strategy")
                C1 = -C1.cpu().numpy()
                C2 = -C2.cpu().numpy()
                weight = torch.FloatTensor(
                    ot.gromov.gromov_wasserstein(C1, C2, a, b, 'kl_loss'))
            else:
                weight = torch.FloatTensor(ot.emd(a, b, M))

    return weight.to(device)


def optimal_transport_align(X: torch.Tensor,
                            Y: torch.Tensor,
                            device: torch.device,
                            reg: float = 0.1,
                            reg_m: float = 0.01,
                            ot_strategy: str = "ub_sinkhorn",
                            requires_grad: bool = False) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Align embeddings using optimal transport with optional gradient computation.

    Args:
        X: Source embeddings (batch_size x n x d)
        Y: Target embeddings (batch_size x m x d)
        device: PyTorch device
        reg: Entropic regularization parameter
        reg_m: Marginal regularization parameter
        ot_strategy: OT strategy to use
        requires_grad: Whether to compute gradients

    Returns:
        Tuple of (mean cosine similarity, aligned embeddings, transport matrices)
    """
    batch_size = X.shape[0]
    cos_sims = []
    X_aligned_ot = []
    Ts = []

    for i in range(batch_size):
        X_i = X[i]
        Y_i = Y[i]

        A = pairwise_cosine(X_i, Y_i)
        if ot_strategy == "gw":
            C1 = pairwise_cosine(X_i, X_i)
            C2 = pairwise_cosine(Y_i, Y_i)
            W = optimal_transport_weight(A, device, C1, C2,
                                         ot_strategy="gw",
                                         requires_grad=requires_grad)
        else:
            W = optimal_transport_weight(A, device,
                                         reg=reg,
                                         reg_m=reg_m,
                                         ot_strategy=ot_strategy,
                                         requires_grad=requires_grad)

        T = A * W
        Ts.append(T)

        X_i_aligned = torch.mm(T, X_i)
        if requires_grad:
            cos_sims.append(cos(X_i_aligned, Y_i).mean())
        else:
            cos_sims.append(cos(X_i_aligned, Y_i).mean().detach().cpu().numpy())
        X_aligned_ot.append(X_i_aligned)

    Xs = torch.stack(X_aligned_ot, dim=0)
    Ts_stack = torch.stack(Ts, dim=0)

    if requires_grad:
        X_Y_cos = torch.mean(torch.stack(cos_sims))
    else:
        X_Y_cos = np.mean(cos_sims)

    return X_Y_cos, Xs, Ts_stack


def optimal_transport_align_test(X_test: torch.Tensor,
                                 Y_test: torch.Tensor,
                                 T: torch.Tensor,
                                 requires_grad: bool = False) -> Tuple[float, torch.Tensor]:
    """
    Align test embeddings using pre-computed transport matrix with optional gradients.

    Args:
        X_test: Test source embeddings
        Y_test: Test target embeddings
        T: Pre-computed transport matrix
        requires_grad: Whether to compute gradients

    Returns:
        Tuple of (mean cosine similarity, aligned embeddings)
    """
    cosine_list = []
    x_test_aligned_list = []

    for i in range(X_test.shape[0]):
        x_test_i = X_test[i]
        y_test_i = Y_test[i]

        x_test_aligned_i = torch.mm(T, x_test_i)
        x_test_aligned_list.append(x_test_aligned_i)

        if requires_grad:
            cosine_list.append(cos(x_test_aligned_i, y_test_i).mean())
        else:
            cosine_list.append(cos(x_test_aligned_i, y_test_i).mean().detach().cpu().numpy())

    x_test_aligned = torch.stack(x_test_aligned_list, dim=0)

    if requires_grad:
        mean_cosine = torch.mean(torch.stack(cosine_list))
    else:
        mean_cosine = np.mean(cosine_list)

    return mean_cosine, x_test_aligned