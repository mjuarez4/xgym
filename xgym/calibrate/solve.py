import numpy as np
import torch
from tqdm import tqdm


def make_motion_pairs(poses):
    """Turn absolute poses into relative motion pairs."""
    motions = []
    for i in range(len(poses)):
        for j in range(len(poses)):
            if i == j:
                continue
            A_ij = np.linalg.inv(poses[i]) @ poses[j]
            motions.append(A_ij)
    return motions


def torch_handeye_solver(T_As, T_Bs, lr=1e-4, steps=int(2_000)):
    """
    Solves AX = XB for X using gradient descent in PyTorch.
    T_As, T_Bs: list of N numpy 4x4 matrices (A_i, B_i)
    Returns: torch 4x4 matrix for X
    """

    assert len(T_As) == len(T_Bs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TAs = torch.tensor(np.stack(T_As), dtype=torch.float32, device=device)  # (N, 4, 4)
    TBinvs = np.linalg.inv(T_Bs)
    TBinvs = torch.tensor(np.stack(TBinvs), dtype=torch.float32, device=device)

    A, B = make_motion_pairs(T_As), make_motion_pairs(T_Bs)

    # Convert inputs to torch
    A = torch.tensor(np.stack(A), dtype=torch.float32, device=device)  # (N, 4, 4)
    B = torch.tensor(np.stack(B), dtype=torch.float32, device=device)

    # Initialize X = [R | t] (3x4) as identity
    R = torch.eye(3, device=device).unsqueeze(0).repeat(1, 1, 1)  # (1, 3, 3)
    t = torch.zeros((1, 3, 1), device=device)  # (1, 3, 1)

    R = torch.nn.Parameter(R.clone().detach().requires_grad_())
    t = torch.nn.Parameter(t.clone().detach().requires_grad_())

    optimizer = torch.optim.Adam([R, t], lr=lr)

    for step in tqdm(range(steps)):
        optimizer.zero_grad()

        # Reconstruct full X (N, 4, 4)
        # with torch.no_grad():
        # R.data = project_to_rotation(R.data)
        # R_proj = project_to_rotation(R.data)  # force R to stay on SO(3)

        X = torch.eye(4, device=device)
        # X[:3, :3] = R # .data
        X[:3, 3:] = t[0]
        X = X.unsqueeze(0)

        # Compute residuals: A_i X - X B_i
        lhs = A @ X
        rhs = X @ B
        diff = lhs - rhs
        loss = 0
        # loss = torch.mean((diff[:, :3, :] ** 2))  # only position and rotation part

        # upweight the translation part of loss
        loss += torch.mean(diff[:, :3, 3:] ** 2)

        # compute base frame AiXBi−1=I
        base = TAs @ X @ TBinvs
        # error is pairwise error for all bases base_0-base_N
        # (they should be the same)
        # diff = torch.mean(base.unsqueeze(1) - base.unsqueeze(0) )
        #  loss += diff

        if step % 500 == 0 or step == steps - 1:
            print(f"Step {step}, loss = {loss.item():.6f}")

        loss.backward()
        optimizer.step()

    # Return final X
    # R_final = project_to_rotation(R[0]).detach().cpu()
    t_final = t[0].detach().cpu().numpy()
    X_final = np.eye(4)
    # X_final[:3, :3] = R.detach().cpu()
    X_final[:3, 3] = t_final.squeeze()

    return X_final


def project_to_rotation(R):
    """Projects a 3x3 matrix onto SO(3) using SVD"""
    U, _, Vt = torch.linalg.svd(R)
    R_proj = U @ Vt
    if torch.det(R_proj) < 0:  # fix reflection if necessary
        U[:, -1] *= -1
        R_proj = U @ Vt
    return R_proj
