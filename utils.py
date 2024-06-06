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

@jit
def genrete_realx_gaussian_field_3d_k_mask(randns, dtype=jnp.float64, cdtype=jnp.complex128):
    """
    randns is an array with nc^3 standard normal vars
    """
    #import numpy as jnp    # change to jax after! fixme
    
    randns = randns.flatten()
    nc = round(randns.shape[0]**(1./3))
    
    # due to G*(f) = G(-f) symmetry, the return value will have shape nc,nc/2+1 (assuming d even)
    n = round(nc/2)+1
    
    ret = jnp.zeros(shape=(nc,nc,n,), dtype=cdtype) 
    
    # grid for masking
    #grid0, grid1, grid2 = jnp.meshgrid(jnp.arange(nc), jnp.arange(nc), jnp.arange(n))
     
    # fill all non k2=0 and nyq voxels with complex number  (n-1 is nyq)
    s = (nc, nc, n-2)
    m = math.prod(s)
    
    #mask = (1 <= grid2) * (grid2 < n-1)
    pads = ((0,0), (0,0), (1,1))    # need to pad r such that it has same shape as mask, and aligns with the 1s.  # fixme, can completely remove mask!
    
    i = 0         # running index for taking dofs    for neatness, always have i= above r=
    r = randns[i:i+m].reshape(s)    # do sqrt2 just before end
    i += m
    
    ret += jnp.pad(r, pads)
    
    r = randns[i:i+m].reshape(s) * 1j
    i += m
    
    ret += jnp.pad(r, pads)
    
    # fill k2=0 nyq>k0>0  (note use of n in firt index now) only want to fill half and then will conj flip by sim.
    # we have a choice to reflect in axis 0 or 1. we chose to reflect in 1 first, 
    # so compared to 2d we just have an extra :, at the beginnign.
    # but then we'll have anthertep to reflect in 0 dimension
    # to keep track of rands easily will do reflection conj by hand.
    s = (nc, n-2, 1)
    m = math.prod(s)
    for b in [0,n-1]:                                # same methodology for when grid2=0 or nyq, o loop
        
        r = randns[i:i+m].reshape(s)
        i += m
        
        pads = [None, None, None]   # save space by updating pads[i] depending on line.
        if   b == 0:   pads[2] = (0,n-1)   # pads[2] is the same throughout loop
        elif b == n-1: pads[2] = (n-1,0) 
        
        #mask = (1 <= grid1) * (grid1 < n-1) * (grid2 == b)
        pads[0] = (0,0)
        pads[1] = (1,n-1)
        
        ret += jnp.pad(r, pads)
        
        # do reflections for grid2=0 before shifting randns for imaginary part agin lots of steps: now we will need to pad AND mask
        # padding ofc needed for shape as above, and mask needed because we want to only reflect certain subregions of r each time)
        #pads[1] = (n,0)   # same for all of these reflections
        
        
        # real part reflections. recall r is unpadded, so simple ::-1 (in 1 index) followed by appropriate padding and 0 index works!
        pads[1] = (n,0)   # the 1 axis always wants these pads for the mirror region
        #mask = (grid0 == 0) * (n <= grid1) * (grid2 == b) 
        pads[0] = (0,nc-1)
        ret += jnp.pad(r[0,::-1,b][None,:,None], pads)
        #mask = (grid0 == n-1) * (n <= grid1) * (grid2 == b)
        pads[0] = (n-1,n-2)
        ret += jnp.pad(r[n-1,::-1,b][None,:,None], pads) 
        #mask = (n <= grid0) * (n <= grid1) * (grid2 == b) 
        pads[0] = (n,0)
        ret += jnp.pad(r[n-2:0:-1,::-1,b][:,:,None], pads)
        #mask = (1 <= grid0) * (grid0 < n-1) * (n <= grid1) * (grid2 == b)
        pads[0] = (1,n-1)
        ret += jnp.pad(r[:n-1:-1,::-1,b][:,:,None], pads)
        
        # now do same for imaginary part
        r = randns[i:i+m].reshape(s) * 1j
        i += m
        
        #mask = (1 <= grid1) * (grid1 < n-1) * (grid2 == b)
        pads[0] = (0,0)
        pads[1] = (1,n-1)
        
        ret += jnp.pad(r, pads)
        
        # do reflections for grid2=0 before shifting randns for imaginary part agin lots of steps. - for conj
        pads[1] = (n,0)   # the 1 axis always wants these pads for the mirror region
        #mask = (grid0 == 0) * (n <= grid1) * (grid2 == b)
        pads[0] = (0,nc-1)
        ret -= jnp.pad(r[0,::-1,b][None,:,None], pads)
        #mask = (grid0 == n-1) * (n <= grid1) * (grid2 == b)
        pads[0] = (n-1,n-2)
        ret -= jnp.pad(r[n-1,::-1,b][None,:,None], pads)
        #mask = (n <= grid0) * (n <= grid1) * (grid2 == b)
        pads[0] = (n,0)
        ret -= jnp.pad(r[n-2:0:-1,::-1,b][:,:,None], pads)
        #mask = (1 <= grid0) * (grid0 < n-1) * (n <= grid1) * (grid2 == b)
        pads[0] = (1,n-1)
        ret -= jnp.pad(r[:n-1:-1,::-1,b][:,:,None], pads)
        
    # now we fill the remaining lines across the 0 axis
    s = (n-2, 1, 1)
    m = math.prod(s)
    for b2 in [0, n-1]:
        for b1 in [0, n-1]:
            r = randns[i:i+m].reshape(s)
            i += m
            
            pads = [None, None, None]   # save space by updating pads[i] depending on line.
            if   b2 == 0:   pads[2] = (0,n-1)   # pads[2] is the same throughout loop
            elif b2 == n-1: pads[2] = (n-1,0) 
            if   b1 == 0:   pads[1] = (0,nc-1)   # pads[1] is the same throughout loop
            elif b1 == n-1: pads[1] = (n-1,n-2) 

            # real
            #mask = (1 <= grid0) * (grid0 < n-1) * (grid1 == b1) * (grid2 == b2)
            pads[0] = (1,n-1)
            ret += jnp.pad(r, pads)
            # reflect
            #mask = (n <= grid0) * (grid1 == b1) * (grid2 == b2)
            pads[0] = (n,0)
            ret += jnp.pad(r[::-1, b1, b2][:,None,None], pads)         #n-2:0:-1
            
            
            r = randns[i:i+m].reshape(s) * 1j
            i += m
            
            # im
            #mask = (1 <= grid0) * (grid0 < n-1) * (grid1 == b1) * (grid2 == b2)
            pads[0] = (1,n-1)
            ret += jnp.pad(r, pads)
            # reflect
            #mask = (n <= grid0) * (grid1 == b1) * (grid2 == b2)
            pads[0] = (n,0)
            ret -= jnp.pad(r[::-1, b1, b2][:,None,None], pads)
    
    # divide everyhting by sqrt(2) before doing real components
    ret /= jnp.sqrt(2)
    
    # now fill in real parts
    s = (1, 1, 1)
    m = math.prod(s)
    for b2 in [0, n-1]:
        for b1 in [0, n-1]:
            for b0 in [0, n-1]:
                
                pads = [None, None, None]   # save space by updating pads[i] depending on line.
                if   b2 == 0:   pads[2] = (0,n-1)   # pads[2] is the same throughout loop
                elif b2 == n-1: pads[2] = (n-1,0) 
                if   b1 == 0:   pads[1] = (0,nc-1)   # pads[1] is the same throughout loop
                elif b1 == n-1: pads[1] = (n-1,n-2) 
                if   b0 == 0:   pads[0] = (0,nc-1)   # pads[0] is the same throughout loop
                elif b0 == n-1: pads[0] = (n-1,n-2) 

                r = randns[i:i+m].reshape(s)
                i += m
    
                #mask = (grid0 == b0) * (grid1 == b1) * (grid2 == b2)
                ret += jnp.pad(r, pads)
            
    assert(i == nc**3)
    
    return ret #jnp.asarray(ret)

#@jit   # don't jit, this is based on numpy
def genrete_realx_gaussian_field_3d_k_mask_freqmags(nc, dtype=jnp.float64, cdtype=jnp.complex128):
    """
    let's get the frequencies of each component of the input of the function, for preconsditioning. 
    this is like rfftfreq essentially but the input of genrete_realx_gaussian_field_3d_k_mask
    randns_nc is the nc associated with rands, where there are nc**3 dofs. Assuming 3d.
    
    much easier to dow tih item assignment, so will just compute the freqs using numpy.
    only need to do once at start of sim, so not a big problem.
    
    can remove the actual computation of the grid eventually FIXME
    """
    #import numpy as jnp    # change to jax after! fixme
    
    ###randns = randns.flatten()
    ##nc = round(randns.shape[0]**(1./3))
    randns = jnp.ones(nc**3)   # just a mask of ones. pad with 0s.
    
    # due to G*(f) = G(-f) symmetry, the return value will have shape nc,nc/2+1 (assuming d even)
    n = round(nc/2)+1
    
    
    ###ret_shape = (nc,nc,n,)
    ###ret = jnp.zeros(shape=ret_shape, dtype=cdtype) 
    
    # output freq grid too!! these few lines, along with a freqs_randn[i:i+m] = freqs_ret[jnp.pad(r, pads) != 0].flatten() acompantying in between each i and i+=m. (not this needs to be AFTER the appropriate pads are defined!)
    # use != 0 not astype.bool, because for np 1j = True, but for jax 1j = False 
    ###freq_period = 2. * jnp.pi / spacing
    kvec = rfftnfreq(shape=(nc,)*3, spacing=1, dtype=dtype)   #  spacing=1 means grid units/2pi units, so we'll have to multiply by grid spacing after
    freqs_ret = jnp.sqrt(sum(k**2 for k in kvec))   # this is the freqs corresponding to the returned k-space field. this is just rfft_freq.  
    freqs_randn = np.empty(nc**3)       # this is the k corresponding to each of the rands input. i.e. use numpy for slicing purposes.
    
    # grid for masking
    #grid0, grid1, grid2 = jnp.meshgrid(jnp.arange(nc), jnp.arange(nc), jnp.arange(n))
     
    # fill all non k2=0 and nyq voxels with complex number  (n-1 is nyq)
    s = (nc, nc, n-2)
    m = math.prod(s)
    
    #mask = (1 <= grid2) * (grid2 < n-1)
    pads = ((0,0), (0,0), (1,1))    # need to pad r such that it has same shape as mask, and aligns with the 1s.  # fixme, can completely remove mask!
    
    i = 0         # running index for taking dofs    for neatness, always have i= above r=
    r = randns[i:i+m].reshape(s)    # do sqrt2 just before end
    freqs_randn[i:i+m] = freqs_ret[jnp.pad(r, pads) != 0].flatten() 
    i += m
    
    ###ret += jnp.pad(r, pads)
    
    r = randns[i:i+m].reshape(s) * 1j
    freqs_randn[i:i+m] = freqs_ret[jnp.pad(r, pads) != 0].flatten() 
    i += m
    
    ###ret += jnp.pad(r, pads)
    
    # fill k2=0 nyq>k0>0  (note use of n in firt index now) only want to fill half and then will conj flip by sim.
    # we have a choice to reflect in axis 0 or 1. we chose to reflect in 1 first, 
    # so compared to 2d we just have an extra :, at the beginnign.
    # but then we'll have anthertep to reflect in 0 dimension
    # to keep track of rands easily will do reflection conj by hand.
    s = (nc, n-2, 1)
    m = math.prod(s)
    
    for b in [0,n-1]:                                # same methodology for when grid2=0 or nyq, o loop
        
        pads = [None, None, None]   # save space by updating pads[i] depending on line.
        if   b == 0:   pads[2] = (0,n-1)   # pads[2] is the same throughout loop
        elif b == n-1: pads[2] = (n-1,0) 
        
        #mask = (1 <= grid1) * (grid1 < n-1) * (grid2 == b)
        pads[0] = (0,0)
        pads[1] = (1,n-1)
        
        r = randns[i:i+m].reshape(s)
        freqs_randn[i:i+m] = freqs_ret[jnp.pad(r, pads) != 0].flatten() 
        i += m
        
        ###ret += jnp.pad(r, pads)
        
        # do reflections for grid2=0 before shifting randns for imaginary part agin lots of steps: now we will need to pad AND mask
        # padding ofc needed for shape as above, and mask needed because we want to only reflect certain subregions of r each time)
        #pads[1] = (n,0)   # same for all of these reflections
        
        """
        # real part reflections. recall r is unpadded, so simple ::-1 (in 1 index) followed by appropriate padding and 0 index works!
        pads[1] = (n,0)   # the 1 axis always wants these pads for the mirror region
        #mask = (grid0 == 0) * (n <= grid1) * (grid2 == b) 
        pads[0] = (0,nc-1)
        ret += jnp.pad(r[0,::-1,b][None,:,None], pads)
        #mask = (grid0 == n-1) * (n <= grid1) * (grid2 == b)
        pads[0] = (n-1,n-2)
        ret += jnp.pad(r[n-1,::-1,b][None,:,None], pads) 
        #mask = (n <= grid0) * (n <= grid1) * (grid2 == b) 
        pads[0] = (n,0)
        ret += jnp.pad(r[n-2:0:-1,::-1,b][:,:,None], pads)
        #mask = (1 <= grid0) * (grid0 < n-1) * (n <= grid1) * (grid2 == b)
        pads[0] = (1,n-1)
        ret += jnp.pad(r[:n-1:-1,::-1,b][:,:,None], pads)
        """
               
        #mask = (1 <= grid1) * (grid1 < n-1) * (grid2 == b)
        pads[0] = (0,0)
        pads[1] = (1,n-1)
        
        # now do same for imaginary part
        r = randns[i:i+m].reshape(s) * 1j
        freqs_randn[i:i+m] = freqs_ret[jnp.pad(r, pads) != 0].flatten() 
        i += m
        
        ###ret += jnp.pad(r, pads)
        """
        # do reflections for grid2=0 before shifting randns for imaginary part agin lots of steps. - for conj
        pads[1] = (n,0)   # the 1 axis always wants these pads for the mirror region
        #mask = (grid0 == 0) * (n <= grid1) * (grid2 == b)
        pads[0] = (0,nc-1)
        ret -= jnp.pad(r[0,::-1,b][None,:,None], pads)
        #mask = (grid0 == n-1) * (n <= grid1) * (grid2 == b)
        pads[0] = (n-1,n-2)
        ret -= jnp.pad(r[n-1,::-1,b][None,:,None], pads)
        #mask = (n <= grid0) * (n <= grid1) * (grid2 == b)
        pads[0] = (n,0)
        ret -= jnp.pad(r[n-2:0:-1,::-1,b][:,:,None], pads)
        #mask = (1 <= grid0) * (grid0 < n-1) * (n <= grid1) * (grid2 == b)
        pads[0] = (1,n-1)
        ret -= jnp.pad(r[:n-1:-1,::-1,b][:,:,None], pads)
        """
    # now we fill the remaining lines across the 0 axis
    s = (n-2, 1, 1)
    m = math.prod(s)
    for b2 in [0, n-1]:
        for b1 in [0, n-1]:
            
            pads = [None, None, None]   # save space by updating pads[i] depending on line.
            if   b2 == 0:   pads[2] = (0,n-1)   # pads[2] is the same throughout loop
            elif b2 == n-1: pads[2] = (n-1,0) 
            if   b1 == 0:   pads[1] = (0,nc-1)   # pads[1] is the same throughout loop
            elif b1 == n-1: pads[1] = (n-1,n-2) 

            # real
            #mask = (1 <= grid0) * (grid0 < n-1) * (grid1 == b1) * (grid2 == b2)
            pads[0] = (1,n-1)
            
            r = randns[i:i+m].reshape(s)
            freqs_randn[i:i+m] = freqs_ret[jnp.pad(r, pads) != 0].flatten() 
            i += m
            
            ###ret += jnp.pad(r, pads)
            
            # reflect
            #mask = (n <= grid0) * (grid1 == b1) * (grid2 == b2)
            pads[0] = (n,0)
            ###ret += jnp.pad(r[::-1, b1, b2][:,None,None], pads)         #n-2:0:-1
            
            
            # im
            #mask = (1 <= grid0) * (grid0 < n-1) * (grid1 == b1) * (grid2 == b2)
            pads[0] = (1,n-1)
            
            r = randns[i:i+m].reshape(s) * 1j
            freqs_randn[i:i+m] = freqs_ret[jnp.pad(r, pads) != 0].flatten() 
            i += m
            
            ###ret += jnp.pad(r, pads)
            
            # reflect
            #mask = (n <= grid0) * (grid1 == b1) * (grid2 == b2)
            pads[0] = (n,0)
            ###ret -= jnp.pad(r[::-1, b1, b2][:,None,None], pads)
    
    # divide everyhting by sqrt(2) before doing real components
    ###ret /= jnp.sqrt(2)
    
    # now fill in real parts
    s = (1, 1, 1)
    m = math.prod(s)
    for b2 in [0, n-1]:
        for b1 in [0, n-1]:
            for b0 in [0, n-1]:
                
                pads = [None, None, None]   # save space by updating pads[i] depending on line.
                if   b2 == 0:   pads[2] = (0,n-1)   # pads[2] is the same throughout loop
                elif b2 == n-1: pads[2] = (n-1,0) 
                if   b1 == 0:   pads[1] = (0,nc-1)   # pads[1] is the same throughout loop
                elif b1 == n-1: pads[1] = (n-1,n-2) 
                if   b0 == 0:   pads[0] = (0,nc-1)   # pads[0] is the same throughout loop
                elif b0 == n-1: pads[0] = (n-1,n-2) 

                r = randns[i:i+m].reshape(s)
                freqs_randn[i:i+m] = freqs_ret[jnp.pad(r, pads) != 0].flatten() 
                i += m
    
                #mask = (grid0 == b0) * (grid1 == b1) * (grid2 == b2)
                ###ret += jnp.pad(r, pads)
            
    assert(i == nc**3)
    
    return freqs_randn #jnp.asarray(ret)