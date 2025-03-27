# real space posterior (Lk=0) with gaussian noise

@jit
def compute_res(modes, p0, data, conf):
    if modes.ndim == 1:
        modes = modes.reshape((round(modes.shape[-1]**(1./3)),)*3)
    mesh = evolve(p0, modes, conf)
    res = data.reshape(mesh.shape) - mesh
    res = jnp.fft.rfftn(res, s=conf.ptcl_grid_shape, norm='ortho')
    return res

@jit
def nlog_lik(modes, p0, data, dnoise, conf):
    res = compute_res(modes=modes, p0=p0, data=data, conf=conf)
    dnoise_var = dnoise**2
    ret = 0.5 * jnp.sum(jnp.abs(res)**2)/dnoise_var
    ret /= conf.ptcl_grid_shape[-1]**3
    return  ret

@jit
def nlog_pri(modes, conf):
    ret = 0.5 * jnp.sum(modes**2)
    ret /= conf.ptcl_grid_shape[-1]**3
    return ret

@jit
def nlog_prob(modes, p0, data, dnoise, conf):
    log_lik = -nlog_lik(modes=modes, p0=p0, data=data, dnoise=dnoise, conf=conf)
    log_prior = -nlog_pri(modes, conf)
    log_prob = log_lik + log_prior
    
    return -log_prob
