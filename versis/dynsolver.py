from veros.core.operators import numpy as npx
from veros import veros_routine, veros_kernel, KernelOutput

from versis.parameters import recip_rhoSea, gravity, seaIceLoadFac
from versis.freedrift_solver import freedrift_solver
from versis.evp_solver import evp_solver
from versis.surface_forcing import surface_forcing


@veros_kernel
def calc_SurfaceForcing(state):

    '''calculate surface forcing due to wind and ocean surface tilt'''

    vs = state.variables

    # calculate surface stresses from wind and ice velocities
    tauX, tauY = surface_forcing(state)

    # calculate forcing by surface stress
    WindForcingX = tauX * 0.5 * (vs.Area + npx.roll(vs.Area,1,0))
    WindForcingY = tauY * 0.5 * (vs.Area + npx.roll(vs.Area,1,1))

    # calculate geopotential anomaly. the surface pressure and sea ice load are
    # used as they affect the sea surface height anomaly
    phiSurf = gravity * vs.ssh_an
    if state.settings.useRealFreshWaterFlux:
        phiSurf = phiSurf + (vs.surfPress + vs.SeaIceLoad * gravity * seaIceLoadFac) * recip_rhoSea
    else:
        phiSurf = phiSurf + vs.surfPress * recip_rhoSea

    # add in tilt
    WindForcingX = WindForcingX - vs.SeaIceMassU \
                    * vs.recip_dxC * ( phiSurf - npx.roll(phiSurf,1,0) )
    WindForcingY = WindForcingY - vs.SeaIceMassV \
                    * vs.recip_dyC * ( phiSurf - npx.roll(phiSurf,1,1) )

    return KernelOutput(WindForcingX = WindForcingX,
                        WindForcingY = WindForcingY,
                        tauX         = tauX,
                        tauY         = tauY)

@veros_routine
def update_SurfaceForcing(state):

    '''retrieve surface forcing and update state object'''

    SurfaceForcing = calc_SurfaceForcing(state)
    state.variables.update(SurfaceForcing)

@veros_kernel
def calc_IceVelocities(state):

    '''calculate ice velocities from surface and ocean forcing'''

    sett = state.settings

    if sett.useFreedrift:
        uIce, vIce = freedrift_solver(state)

    if sett.useEVP:
        uIce, vIce = evp_solver(state)

    return KernelOutput(uIce = uIce, vIce = vIce)

@veros_routine
def update_IceVelocities(state):

    '''retrieve ice velocities and update state object'''

    IceVelocities = calc_IceVelocities(state)
    state.variables.update(IceVelocities)