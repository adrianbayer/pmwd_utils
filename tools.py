import numpy as np
import numpy

def fftk(shape, boxsize, symmetric=True, finite=False, dtype=np.float64):
    """ return kvector given a shape (nc, nc, nc) and boxsize """
    k = []
    for d in range(len(shape)):
        kd = numpy.fft.fftfreq(shape[d])
        kd *= 2 * numpy.pi / boxsize * shape[d]
        kdshape = numpy.ones(len(shape), dtype='int')
        if symmetric and d == len(shape) -1:
            kd = kd[:shape[d]//2 + 1]
        kdshape[d] = len(kd)
        kd = kd.reshape(kdshape)

        k.append(kd.astype(dtype))
    del kd, kdshape
    return k

def power(f1, f2=None, boxsize=1.0, kmin=0, kmax=None, dk=None, symmetric=True, demean=True, Nmu=1, los=[0,0,1]):
    """
    f1: delta+1 field
    f2: for cross power specify second delta+1 field
    boxsize: in mpc/h
    kmin, kmax, dk: k binning in h/mpc
    symmetric: whether to assume rfft. keep true
    demean: make mean of field of 1 (if user input delta rather than delta+1)
    Nmu: number of mu bins
    los: line of sight vector
    """
    if demean and abs(f1.mean()) < 1e-3:
        print('Add 1 to get nonzero mean of %0.3e'%f1.mean())
        f1 = f1*1 + 1
    if demean and f2 is not None:
        if abs(f2.mean()) < 1e-3:
            print('Add 1 to get nonzero mean of %0.3e'%f2.mean())
            f2 =f2*1 + 1

    if symmetric: c1 = numpy.fft.rfftn(f1)#, s=f1shape)
    else: c1 = numpy.fft.fftn(f1)#, s=f1shape)
    if demean : c1 /= c1[0, 0, 0].real
    c1[0, 0, 0] = 0
    if f2 is not None:
        if symmetric: c2 = numpy.fft.rfftn(f2)#, s=f1shape)
        else: c2 = numpy.fft.fftn(f2)#, s=f1shape)
        if demean : c2 /= c2[0, 0, 0].real
        c2[0, 0, 0] = 0
    else:
        c2 = c1
    #x = (c1 * c2.conjugate()).real
    x = c1.real* c2.real + c1.imag*c2.imag
    del c1
    del c2
    
    kk = fftk(f1.shape, boxsize, symmetric=symmetric)
    k = sum(kkk**2 for kkk in kk)**0.5
    
    kF = 2 * np.pi / boxsize  # fundamental 1d freq
    if dk is None:
        dk = kF
        
    if kmax is None:
        kmax1d = kF * f1.shape[-1] / 2
        #kmax3d = np.sqrt(3) * kmax1d
        kmax = kmax1d + dk / 2    # dk/2 ensures we get to kmax (which is a mutiple of dk by defn)

    kedges = np.arange(kmin, kmax, dk)

    if Nmu > 1:
        los = np.array(los, dtype=bool)
        #kper = sum(kkk**2 for i,kkk in enumerate(kk) if los[i] == 0)**0.5 
        kpar = sum(kkk**2 for i,kkk in enumerate(kk) if los[i] == 1)**0.5 
        mu = kpar / k
        mubinedges = np.linspace(0, 1, Nmu+1)   # unlike nbodykit, bin from mu=0
        mubincenters = (mubinedges[1:] + mubinedges[:-1]) / 2
        powers = []

        for nmu in range(Nmu):
        
            mask = (mubinedges[nmu] <= mu) * (mu < mubinedges[nmu+1])
            if nmu == Nmu - 1:   # i.e. maximum mu bin. need to include mu=1 information too, but don't want <= = or we'd double count.
                assert(mubinedges[nmu+1] == 1)
                mask = mask + (mu == mubinedges[nmu+1])
            
            
            H, edges = numpy.histogram(k[mask].flat, weights=x[mask].flat, bins=kedges) 
            N, edges = numpy.histogram(k[mask].flat, bins=edges)
            
            xsum, Nsum = np.zeros(len(kedges) + 1), np.zeros(len(kedges) + 1)
            dig = np.digitize(k[mask].flat, edges)
            xsum.flat += np.bincount(dig, weights=k[mask].flat, minlength=xsum.size)
            Nsum.flat += np.bincount(dig, minlength=xsum.size)
            
            center= edges[1:] + edges[:-1]
            center *= 0.5
            #print(0.5*center, (xsum/Nsum)[1:-1])
            if 1:#dk is not None:
                center = (xsum/Nsum)[1:-1]
            
            power = H *boxsize**3 / N
            power[power == 0] = np.NaN
            powers.append(power)
            
        return center, powers, mubincenters

    else:
        H, edges = numpy.histogram(k.flat, weights=x.flat, bins=kedges) 
        N, edges = numpy.histogram(k.flat, bins=edges)
        
        xsum, Nsum = np.zeros(len(kedges) + 1), np.zeros(len(kedges) + 1)
        dig = np.digitize(k.flat, edges)
        xsum.flat += np.bincount(dig, weights=k.flat, minlength=xsum.size)
        Nsum.flat += np.bincount(dig, minlength=xsum.size)
        #print(N,Nsum)
        center= edges[1:] + edges[:-1]
        center *= 0.5
        #print(0.5*center, (xsum/Nsum)[1:-1])
        if 1:#dk is not None:
            center = (xsum/Nsum)[1:-1]
            
        power = H *boxsize**3 / N
        power[power == 0] = np.NaN
        return center,  power

def power_pyl3(f1, f2=None, boxsize=1, MAS = None, kmin=0, kmax=None, dk=None, symmetric=True, demean=True, eps=1e-9, Nmu=1, los=[0,0,1]):
    """
    all args after MAS are meaningless, but help make code transferable FIXME
    """
    import Pk_library as PKL
    print('using power_pyl3')
    if f2 is None:
        delta = (f1-1).astype(np.float32)
        Pk = PKL.Pk(delta=delta, BoxSize=boxsize, axis=0, MAS=MAS)#, threads, verbose)
        k, P = Pk.k3D, Pk.Pk[:,0]
    else:
        delta1 = (f1-1).astype(np.float32)
        delta2 = (f2-1).astype(np.float32)
        Pk = PKL.XPk([delta1,delta2], BoxSize=boxsize, axis=0, MAS=[MAS,MAS])#, threads)
        k, P = Pk.k3D, Pk.XPk[:,0,0]
        
    return k, P
