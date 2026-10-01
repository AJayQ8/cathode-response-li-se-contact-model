"""Exact elimination of transport4's periodic square finite-volume bulk."""
from functools import lru_cache
import numpy as np
from scipy.linalg import cho_factor, cho_solve, circulant, solve_banded


@lru_cache(maxsize=4)
def boundary_operator(n):
    # H=Lx=kappa=1; dx=dy=1/n. The vertical zero-flux operator
    # has endpoint diagonal 1, interior diagonal 2, and off-diagonal -1.
    z=np.zeros(n)
    diagonal=np.full(n,2.)
    diagonal[[0,-1]]=1.
    forcing=np.zeros(n)
    forcing[0]=1.
    for k in range(1,n//2+1):
        horizontal=4*np.sin(np.pi*k/n)**2
        bands=np.zeros((3,n))
        bands[0,1:]=-1.
        bands[1]=diagonal+horizontal
        bands[2,:-1]=-1.
        column=solve_banded((1,1),bands,forcing,check_finite=False)
        z[k]=column[0]/n+1/(2*n)
        z[-k]=z[k]
    kernel=np.fft.ifft(z).real
    matrix=circulant(kernel)
    matrix=(matrix+matrix.T)/2
    matrix.setflags(write=False)
    return matrix


def solve(active, current, beta):
    a=np.asarray(active,dtype=float)
    assert a.ndim==1 and np.all((a>=0)&(a<=1)) and beta>0
    if np.mean(a)<=0:
        return {'status':'disconnected'}, None, None
    n=len(a)
    z=boundary_operator(n)
    b=np.sqrt(beta*a)
    matrix=(b[:,None]*z)*b[None,:]
    matrix[np.diag_indices(n)]+=1.
    factors=cho_factor(matrix,lower=True,check_finite=False)
    w=cho_solve(factors,b,check_finite=False)
    multiplier=float(n/np.dot(b,w))
    unit_q=multiplier*b*w
    unit_surface=multiplier-z@unit_q
    q=current*unit_q
    active_q=np.where(a>0,current*beta*unit_surface,0.)
    resistance=1+multiplier
    interface=float(np.sum(np.divide(unit_q*unit_q,beta*a,out=np.zeros(n),where=a>0))/n)
    bulk=1+float(np.dot(unit_q,z@unit_q))/n
    residual=unit_q-beta*a*unit_surface
    diagnostics={'status':'solved','resistance_over_bulk':resistance,
                 'current_relative_error':abs(float(np.mean(unit_q))-1),
                 'power_relative_error':abs(interface+bulk-resistance)/resistance,
                 'boundary_law_relative_l2':float(np.linalg.norm(residual)/max(np.linalg.norm(unit_q),1e-30)),
                 'minimum_current_over_applied':float(np.min(unit_q)),
                 'maximum_current_over_applied':float(np.max(unit_q)),
                 'maximum_active_current_over_applied':float(np.max(np.where(a>0,beta*unit_surface,0.))),
                 'zero_gap_current':bool(np.all(q[a==0]==0)),
                 'mean_active_fraction':float(np.mean(a))}
    return diagnostics,q,active_q
