import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sys, os

import jax
from jax import jit, custom_vjp, ensure_compile_time_eval, grad
import jax.numpy as jnp

from pmwd_imports import *

@jit
def evolve(p0, modes, conf, rsd):
    """
    Given initial whitened modes, this function evolves to the final mesh.
    """
    #if modes.ndim == 1:   # this should only happen when real modes input. but maybe ugly so fixme.
    #    modes = modes.reshape((round(modes.shape[-1]**(1./3)),)*3)  # these will be gaussian random vars in x space. evolve will treat each appropriately (pmwd internals)
    #try:
    omegam, sigma_8 = p0
    #except ValueError:   # fixme... check if dict
    #    (omegam, sigma_8) = (p0["Om"], p0["s8"])
    cosmo = Cosmology.from_sigma8(conf, Omega_m=omegam, sigma8=sigma_8, n_s=0.96, Omega_b=0.05, h=0.7)
    cosmo = boltzmann(cosmo, conf)
    lin_modes_c = linear_modes(modes, cosmo, conf, None, False)   # None for `a`, False for `real`. jax bug so need to write like this
    if conf.lpt_order == 0:
        mesh = jnp.fft.irfftn(lin_modes_c, s=conf.ptcl_grid_shape, norm='ortho')
        mesh = normalize_lin_modes(mesh, cosmo, conf)
    else:
        ptcl = simulate_nbody_ptcl(lin_modes_c, cosmo, conf, rsd)
        mesh = scatter(ptcl, conf)

    return mesh

@jit
def simulate_nbody_ptcl(lin_modes_c, cosmo, conf, rsd):
    '''Run LPT simulation (stop at particle info) without evaluating growth & tranfer function               
    '''
    #print("nbody")
    ptcl, obsvbl = lpt(lin_modes_c, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    ptcl = ptcl.replace(disp = ptcl.disp + rsd * ptcl.vel / (conf.a_stop * (100*cosmo.h)**0)) # FIXME check correct resd scaling
    return ptcl

@jit
def simulate_nbody(lin_modes_c, cosmo, conf, rsd):
    '''Run LPT simulation (and paint particles to mesh) without evaluating growth & tranfer function               
    '''
    #print("nbody")
    ptcl = simulate_nbody_ptcl(lin_modes_c, cosmo, conf, rsd)
    mesh = scatter(ptcl, conf)
    return mesh

@jit
def normalize_lin_modes(lin_modes, cosmo, conf):
    """
    taking irfft of lin_modes_c is has weird scaling and is at wrong redshift.
    apply scaling to lin_modes_r to get to delta+1 at final redshit
    FIXME might want to make the scale factor output an argument of this, rather than assume a_stop
    """
    lin_modes /= conf.box_vol**0.5
    lin_modes /= conf.ptcl_cell_vol**0.5   # these converts to same units as pmwd linear_power func
    # linear modes computed at a=None (matter dom), but we want a=1, so will now scale by growth.
    # Note LPT later intrinsically assumes a=None, so don't want to change the a of the linear modes.
    lin_modes *= growth(a=conf.a_stop, cosmo=cosmo, conf=conf, order=1, deriv=0)     # evolve form "a=None" until today
    lin_modes += 1
    
    return lin_modes

@jit
def unnormalize_lin_modes(lin_modes, cosmo, conf):
    """
    taking irfft of lin_modes_c is has weird scaling and is at wrong redshift.
    apply scaling to lin_modes_r to get to delta+1 at final redshit
    FIXME might want to make the scale factor output an argument of this, rather than assume a_stop
    """
    lin_modes -= 1
    lin_modes /= growth(a=conf.a_stop, cosmo=cosmo, conf=conf, order=1, deriv=0)     # evolve form "a=None" until today
    lin_modes *= conf.box_vol**0.5
    lin_modes *= conf.ptcl_cell_vol**0.5   # these converts to same units as pmwd linear_power func
    # linear modes computed at a=None (matter dom), but we want a=1, so will now scale by growth.
    # Note LPT later intrinsically assumes a=None, so don't want to change the a of the linear modes.
    
    return lin_modes

def gendata(conf, seed, cosmo, dnoise=1., rsd=0, savepath=None):
    """
    there;s some duplication of evolve in here, but we want to 
    keep track of the linear modes etc, so rewrite evth again.
    also note this takes cosmo not p0
    return ptcl too. this is fine as gendata is only used to gen dtaa at start.
    evolve will not ouput ptcl.
    """
    modes = white_noise(seed, conf, real=True)
    lin_modes_c = linear_modes(modes, cosmo, conf, None, False)
    lin_modes = jnp.fft.irfftn(lin_modes_c, s=conf.ptcl_grid_shape, norm='ortho')
    lin_modes = normalize_lin_modes(lin_modes, cosmo, conf)
    if conf.lpt_order == 0:
        dens_h = dens_m = lin_modes
        ptcl = None
    else:
        ptcl = simulate_nbody_ptcl(lin_modes_c, cosmo, conf, rsd)
        dens = scatter(ptcl, conf)
        
    np.random.seed(seed*12+45)   # make the noise added to data deterministic based on seed too
    noise = np.random.normal(0, dnoise, dens.size).reshape(dens.shape)
    data = dens + noise
    
    if savepath is not None:
        np.save(savepath + "modes", modes)
        np.save(savepath + "linc", lin_modes_c)
        np.save(savepath + "lin", lin_modes)
        np.save(savepath + "dens", dens)
        np.save(savepath + "data", data)
    
        plotdata(conf, modes, lin_modes, dens, data, savepath)
    
    return modes, lin_modes_c, lin_modes, dens, data, ptcl

def plotdata(conf, modes, lin_modes, dens, data, savepath=None, kmin=0, kmax=None, dk=None):  # fixme i dont think u want +1s except maybe on modes/
    
    box_size = conf.box_size[0]
    k, pk = power_spectrum(1+modes, boxsize=box_size, kmin=kmin, kmax=kmax, dk=dk)
    print(k, pk)
    plt.plot(k, pk, label='modes')
    k, pk = power_spectrum(1+lin_modes, boxsize=box_size, kmin=kmin, kmax=kmax, dk=dk)
    ###############plt.plot(k, pk, label='linear')
    k, pk = power_spectrum(dens, boxsize=box_size, kmin=kmin, kmax=kmax, dk=dk)
    plt.plot(k, pk, label='final')
    k, pk = power_spectrum(data, boxsize=box_size, kmin=kmin, kmax=kmax, dk=dk)
    plt.plot(k, pk, label='data')
    k, pk = power_spectrum(1+data-dens, boxsize=box_size, kmin=kmin, kmax=kmax, dk=dk)
    plt.plot(k, pk, label='noise')
    """ need to add cosmo to arge to make this work
    Plin = linear_power(k, 1, cosmo, conf) 
    fprecon = jnp.sqrt(Plin + 0.1**2 * conf.ptcl_spacing**3)
    plt.plot(k, fprecon**2, label='precon**2')
    plt.plot(k, Plin, label='Plin')
    plt.plot(k, 0.1**2 * conf.ptcl_spacing**3*jnp.ones_like(k), label='0.1 * conf.ptcl_spacing**3')
    """
    plt.xlabel('k (h/Mpc)')
    plt.ylabel('P(k)')
    plt.legend()
    plt.loglog()
    plt.grid(which='both', alpha=0.5)
    plt.savefig(savepath + 'dataps.png')
    plt.close()

        
    #
    fig, axar = plt.subplots(2, 2, figsize=(8, 8))
    im = axar[0, 0].imshow(modes.sum(axis=0))
    plt.colorbar(im, ax=axar[0, 0])
    axar[0, 0].set_title('Modes')
    #im = axar[0, 1].imshow(lin_modes.sum(axis=0))
    #plt.colorbar(im, ax=axar[0, 1])
    #axar[0, 1].set_title('Linear')
    im = axar[1, 0].imshow(dens.sum(axis=0))
    plt.colorbar(im, ax=axar[1, 0])
    axar[1, 0].set_title('Final')
    im = axar[1, 1].imshow(data.sum(axis=0))
    plt.colorbar(im, ax=axar[1, 1])
    axar[1, 1].set_title('Data')
    plt.savefig(savepath + 'dataim.png')
    plt.close()
