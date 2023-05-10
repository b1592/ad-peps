# Set True to apply the apply_permute decorator,
# which applies perform_permute to all SymTensor
# outputs (necessary for gradcheck)
always_permute = True

# Set True to check at every mult call whether
# the contracted legs have conjugate charges
check_mult_inds = False

# In diag_inv, all elements larger than this
# value will be inverted. Normally this will
# be strictly zero, for example when inverting
# the singular values after an svd, but it needs
# to be finite for numerical gradient checking
diag_inv_zero_tol = 0

# Perform consistency check on each new tensor
# This has a large overhead to make sure to
# disable this in actual simulations
do_consistency_checks = False

# Do backward svd pass based on the full tensors
# (before any truncation)
# For this it's required to save the u,s,v tensors
# so it is more expensive
do_precise_svd_backward = True

# The inverse used in the backward svd may be unstable
# in the presence of (nearly) degenerate singular values
# Set this option to a value larger than 0 (usually 1e-12
# should be ok) to use the 'safe inverse' instead of a
# regular inverse
safe_inv_epsilon = 0

# Enables an extra sign-fixing step in the svd, to ensure
# a consistent convention for the sign choices of the singular
# vectors
svd_fix_sign = False

# Clears the stored objects in the ctx during the backward
# step - disable this when you run the backward step with
# retain_graph = True (e.g. for higher-order derivatives)
backward_clear_ctx = True


# Debugging feature to place breaking points in the code
break_now = False


def from_dict(cfg):
    cfg_vars = globals()
    for name, value in cfg.items():
        if name in cfg_vars.keys():
            cfg_vars[name] = value
        else:
            raise ValueError(f"Option {name} = {value} not defined in SymTensor config")
