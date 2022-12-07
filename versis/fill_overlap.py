from veros.core.operators import update, at
from veros import veros_kernel


@veros_kernel
def fill_overlap(state, A):

    sett = state.settings

    A = update(A, at[:sett.olx,:], A[-2*sett.olx:-sett.olx,:])
    A = update(A, at[-sett.olx:,:], A[sett.olx:2*sett.olx,:])
    A = update(A, at[:,:sett.oly], A[:,-2*sett.oly:-sett.oly])
    A = update(A, at[:,-sett.oly:], A[:,sett.oly:2*sett.oly])

    return A

@veros_kernel
def fill_overlap3d(state, A):

    sett = state.settings

    A = update(A, at[:,:sett.olx,:], A[:,-2*sett.olx:-sett.olx,:])
    A = update(A, at[:,-sett.olx:,:], A[:,sett.olx:2*sett.olx,:])
    A = update(A, at[:,:,:sett.oly], A[:,:,-2*sett.oly:-sett.oly])
    A = update(A, at[:,:,-sett.oly:], A[:,:,sett.oly:2*sett.oly])

    return A

@veros_kernel
def fill_overlap_uv(state, U, V):
    return fill_overlap(state, U), fill_overlap(state, V)
