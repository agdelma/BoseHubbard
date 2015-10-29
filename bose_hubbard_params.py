"""

Description:
  Determines t/E_R and U/E_R for the Bose-Hubbard model

Usage:
    bose_hubbard_params.py [-N <N> -V <V> -a <a>] 
    bose_hubbard_params.py -h | --help 

Options:
  -h, --help     Show this screen.
  -N <N>         Number of lattice sites [default: 10].
  -V <V>         Strength of the lattice potential in units of E_R [default: 1.0].
  -a <a>         Ratio of 1D scattering length to lattice spacing [default: 1.0].
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
from scipy.sparse import diags
from scipy.integrate import quad
from docopt import docopt

# -------------------------------------------------------------------------
def build_Hamiltonian(V,q,M):
    '''The MxM tridiagonal matrix.
    
       V is in units of E_R.
       q is in units of k=\pi/a_0'''
    
    d = np.array([(2.0*j+q)**2 + 0.5*V for j in range(-int(M/2),int(M/2))])
    od = -0.25*V*np.ones(len(d)-1)
    H = diags([d,od,od],[0, -1, 1]).toarray()
    return H

# -------------------------------------------------------------------------
def get_spectrum(V,N,M=50,k=3):
    '''Get the spectrum for a range of q/k values.'''
    q = np.arange(-1.0,1.0,2.0/N)
    Eq = np.zeros([N,k])
    C = np.zeros([M,N])
    for i,cq in enumerate(q):
        H = build_Hamiltonian(V,cq,M) 
        e,v = linalg.eigh(H)
        idx = e.argsort()
        Eq[i,:] = e[idx][:k]
        C[:,i] = v[:,idx][:,0]
    return q,Eq,C

# -------------------------------------------------------------------------
def Wannier(x,j,C):
    '''Compute the dimensionless Wannier function at point x/a_0.'''
    M,N = C.shape
    q = np.pi*np.arange(-1.0,1.0,2.0/N)
    l = 2.0*np.pi*np.arange(-int(M/2),int(M/2),1)
    
    Q,L = np.meshgrid(q,l)
    
    w = np.sum(C*np.exp(1j*L*x)*np.exp(1j*Q*x)*np.exp(-1j*Q*j))
    return w/N

# -------------------------------------------------------------------------
def D2Wannier(x,j,C):
    '''The second derivative of the dimensionless Wannier function at point x/a_0.'''
    M,N = C.shape
    q = np.pi*np.arange(-1.0,1.0,2.0/N)
    l = 2.0*np.pi*np.arange(-int(M/2),int(M/2),1)
    
    Q,L = np.meshgrid(q,l)
    
    w = np.sum((Q+L)**2*C*np.exp(1j*L*x)*np.exp(1j*Q*x)*np.exp(-1j*Q*j))
    return w/N

# -------------------------------------------------------------------------
def tKernel(x,V,C):
    '''The kernel of the t/E_R intergral.'''
    w0 = np.conj(Wannier(x,0,C))
    w1 = Wannier(x,1,C)
    w2 = D2Wannier(x,1,C)
    return -np.real(w0*w2/(np.pi*np.pi) + V*w0*w1*(np.sin(np.pi*x))**2)

# -------------------------------------------------------------------------
def get_t(V,N,C):
    ''' compute t/E_R.'''
    return quad(tKernel,-0.5*N,0.5*N,args=(V,C))[0]

# -------------------------------------------------------------------------
def UKerneldelta(x,C):
    '''The kernel of the U/E_R intergral.'''
    w = Wannier(x,0,C)
    return np.real((w*np.conj(w)))**2

# -------------------------------------------------------------------------
def get_Udelta(a,N,C):
    ''' compute U/E_R where a = |a_{1D}/a_0|.'''
    return 4.0/(a*np.pi**2)*quad(UKerneldelta,-0.5*N,0.5*N,args=(C))[0]

# -----------------------------------------------------------------------------
# Begin Main Program 
# -----------------------------------------------------------------------------
def main(): 

    # parse the command line options
    args = docopt(__doc__)

    N = int(args['-N'])
    V = float(args['-V'])
    a = float(args['-a'])

    # Diagonalize the matrix and get the eigenvectors
    q,Eq,C = get_spectrum(V,N,k=1)

    # compute t and U
    t = get_t(V,N,C)
    U = get_Udelta(a,N,C)

    # output the result to the terminal
    print('\nBose-Hubbard Parameters')
    print('=======================')
    print('#%7s %8s %8s %8s %8s %8s' % ('N','a1D/a0','V/E_R','t/E_R','U/E_R','U/t'))
    print('%8d %8.4f %8.4f %8.4f %8.4f %8.4f' % (N,a,V,t,U,U/t))

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__": 
    main()
