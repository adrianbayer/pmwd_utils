import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from utils import gendata
from pmwd_imports import *

seed = 0
nc = 128
boxsize = 400
cell_size = boxsize / nc
order = 2

a_start = 0.1
a_nbody_maxstep = 1
if a_nbody_maxstep > 1 - a_start:
    a_start = 1.
dnoise = .1

float_dtype = jnp.float64

conf = Configuration(ptcl_spacing=cell_size, ptcl_grid_shape=(nc,)*3, lpt_order=order, float_dtype=float_dtype, \
                     a_start=a_start, a_nbody_maxstep=a_nbody_maxstep)

omegam, sigma_8 = 0.3, 0.8
p0 = jnp.array([omegam, sigma_8])
cosmo = Cosmology.from_sigma8(conf, Omega_m=omegam, sigma8=sigma_8, n_s=0.96, Omega_b=0.05, h=0.7)
cosmo = boltzmann(cosmo, conf)

print(gendata(conf, seed, cosmo, dnoise=dnoise, savepath=None))
modes, lin_modes_c, lin_modes, dens, data, ptcl = gendata(conf, seed, cosmo, dnoise=dnoise, savepath=None)
