import jax.numpy as np
from jax import custom_vjp
import jax

def reshape(m, left_ixs, right_ixs):
    m = np.transpose(m, [*left_ixs, *right_ixs])
    left_size = np.prod(np.array([m.shape[i] for i in range(len(left_ixs))]))
    return np.reshape(m, (left_size, -1))

def svd(m, n, mode, cutoff=1e-12, use_iter=False):
    """ Performs an svd with a cutoff
        
        Parameters:
        m (2-D array): matrix to be svd'ed
        n (int):          maximum number of singular values
        mode (str):       absorb the singular values either in
            u ('l'), in v ('r') or not (otherwise)

        Returns:
        torch.Tensor: u
        torch.Tensor: s
        torch.Tensor: v
        such that u*s*v approximates m

    """
    if m.size == 0:
        u = np.zeros([0,0])
        s = np.tensor([])
        v = np.zeros([0,0])
        return u,s,v

    # m = reshape(m, left_ixs, right_ixs)

    if use_iter and n < min(m.size()) and min(m.size()) > 400:
        r
        u, s, v = np.svd(m)
    else:
        # u, s, v = np.linalg.svd(m, full_matrices=False)
        u, s, v = _svd_impl(m)
        ix = np.argsort(s)[::-1]
        s = s[ix]
        u = u[:,ix]
        v = v[ix,:]
            
    # n_above_cutoff = len(np.where(s/s.max() > cutoff)[0])
    # n = min(n, n_above_cutoff)
    # if s.shape[0] > n:
    #     u = u[:,:n]
    #     s = s[:n]
    #     v = v[:n,:]

    u,s,v = _cutoff_matrices(u,s,v,cutoff,n)

    s = np.diag(s)

    # try:
    #     if config.svd_fix_sign:
    #         u, v = _svd_flip(u, v)
    # except:
    #     pass

    # if mode == 'l':
    #     u = u @ s
    # if mode == 'r':
    #     v = s @ v

    return u,s,v

def _cutoff_matrices(u,s,v,cutoff,n):
    # n_above_cutoff = len(np.where(s/s.max() > cutoff)[0])
    n_above_cutoff = np.count_nonzero(s/np.max(s) > cutoff)
    # n_above_cutoff = 8
    n = np.min(np.array([n, n_above_cutoff]))
    if n < np.inf and s.shape[0] > n:
        n = int(n)
        u = u[:,:n]
        s = s[:n]
        v = v[:n,:]
    return u,s,v

@custom_vjp
def _svd_impl(m):
    u, s, v = np.linalg.svd(m, full_matrices=False)
    return u, s, v

def _svd_impl_fwd(m):
    u, s, v = _svd_impl(m)
    # if s.shape[0] < min(m.shape[0],m.shape[1]):
    #     print('SVD def')
    return (u, s, v), (u,s,v)

def _svd_impl_bwd(res, g):
    U, S, V = res
    dU, dS, dV = g
    Vt = V
    V = V.T
    dV = dV.T
    Ut = U.T
    M = U.shape[0]
    N = V.shape[0]
    NS = len(S)

    # F = (S - S[:, None])
    # F = __safe_inverse(F)
    # F.diagonal().fill_(0)

    F = (S - S[:, None])
    F = __safe_inverse(F)
    F = jax.ops.index_update(F, np.diag_indices(F.shape[0]), 0)

    # G = (S + S[:, None])
    # G.diagonal().fill_(np.inf)
    # G = 1/G 

    G = (S + S[:, None])
    G = 1/G 
    G = jax.ops.index_update(G, np.diag_indices(G.shape[0]), 0)

    UdU = Ut @ dU
    VdV = Vt @ dV

    Su = (F+G)*(UdU-UdU.T)/2
    Sv = (F-G)*(VdV-VdV.T)/2

    dA = U @ (Su + Sv + np.diag(dS)) @ Vt 
    if (M>NS):
        # dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt 
        dA = dA + (np.eye(M, dtype=dU.dtype) - U@Ut) @ (dU/S) @ Vt 
    if (N>NS):
        # dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        dA = dA + (U/S) @ dV.T @ (np.eye(N, dtype=dU.dtype) - V@Vt)
    return (dA,)

_svd_impl.defvjp(_svd_impl_fwd, _svd_impl_bwd)

def __safe_inverse(x, epsilon=1e-12):
    if epsilon is None:
        epsilon = config.safe_inv_epsilon
    if epsilon == 0:
        return 1/x
    else:
        return x/(x**2 + epsilon)


def diag_inv(m):
    return np.diag(1 / np.diag(m))
