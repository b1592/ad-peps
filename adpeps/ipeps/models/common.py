import jax.numpy as np

sigmap  = np.array([[0,1],[0,0]])
sigmam  = np.array([[0,0],[1,0]])
sigmaz  = np.array([[1,0],[0,-1]])

id2     = np.array([[1,0],[0,1]])
id4     = np.eye(4)
nsite   = np.array([[0,0],[0,1]])

nup     = np.array([
        [0,0,0,0],
        [0,1,0,0],
        [0,0,0,0],
        [0,0,0,1]
    ])
ndown   = np.array([
        [0,0,0,0],
        [0,0,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])
nupdown = np.array([
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,1]
    ])
