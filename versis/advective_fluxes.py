from veros.core.operators import numpy as npx
from veros.core.operators import update, at
from veros import veros_kernel

from versis.parameters import CrMax
from versis.fill_overlap import fill_overlap


@veros_kernel
def limiter(Cr):

    #return 0       (upwind)
    #return 1       (Lax-Wendroff)
    #return np.max((0, np.min((1, Cr))))    (Min-Mod)
    return npx.maximum(0, npx.maximum(npx.minimum(1,2*Cr), npx.minimum(2,Cr)))

@veros_kernel
def calc_ZonalFlux(state, field, uTrans):

    '''calculate the zonal advective flux using the second order flux limiter method'''

    maskLocW = state.variables.iceMaskU * state.variables.maskInU

    # CFL number of zonal flow
    uCFL = npx.abs(state.variables.uIce * state.settings.deltatTherm * state.variables.recip_dxC)

    # calculate slope ratio Cr
    Rjp = (field[3:,:] - field[2:-1,:]) * maskLocW[3:,:]
    Rj = (field[2:-1,:] - field[1:-2,:]) * maskLocW[2:-1,:]
    Rjm = (field[1:-2,:] - field[:-3,:]) * maskLocW[1:-2,:]

    Cr = npx.where(uTrans[2:-1,:] > 0, Rjm, Rjp)
    Cr = npx.where(npx.abs(Rj) * CrMax > npx.abs(Cr),
                        Cr / Rj, npx.sign(Cr) * CrMax * npx.sign(Rj))
    Cr = limiter(Cr)

    # zonal advective flux for the given field
    ZonalFlux = npx.zeros_like(state.variables.iceMask)
    ZonalFlux = update(ZonalFlux, at[2:-1,:], uTrans[2:-1,:] * (
                field[2:-1,:] + field[1:-2,:]) * 0.5
                - npx.abs(uTrans[2:-1,:]) * ((1 - Cr)
                + uCFL[2:-1,:] * Cr ) * Rj * 0.5)
    ZonalFlux = fill_overlap(state, ZonalFlux)

    return ZonalFlux

@veros_kernel
def calc_MeridionalFlux(state, field, vTrans):

    '''calculate the meridional advective flux using the second order flux limiter method'''

    maskLocS = state.variables.iceMaskV * state.variables.maskInV

    # CFL number of meridional flow
    vCFL = npx.abs(state.variables.vIce * state.settings.deltatTherm * state.variables.recip_dyC)

    # calculate slope ratio Cr
    Rjp = (field[:,3:] - field[:,2:-1]) * maskLocS[:,3:]
    Rj = (field[:,2:-1] - field[:,1:-2]) * maskLocS[:,2:-1]
    Rjm = (field[:,1:-2] - field[:,:-3]) * maskLocS[:,1:-2]

    Cr = npx.where(vTrans[:,2:-1] > 0, Rjm, Rjp)
    Cr = npx.where(npx.abs(Rj) * CrMax > npx.abs(Cr),
                    Cr / Rj, npx.sign(Cr) * CrMax * npx.sign(Rj))
    Cr = limiter(Cr)

    # meridional advective flux for the given field
    MeridionalFlux = npx.zeros_like(state.variables.iceMask)
    MeridionalFlux = update(MeridionalFlux, at[:,2:-1], vTrans[:,2:-1] * (
                field[:,2:-1] + field[:,1:-2]) * 0.5
                - npx.abs(vTrans[:,2:-1]) * ((1 - Cr)
                + vCFL[:,2:-1] * Cr ) * Rj * 0.5)
    MeridionalFlux = fill_overlap(state, MeridionalFlux)

    return MeridionalFlux