from veros.core.operators import numpy as npx
from veros import veros_kernel, KernelOutput, veros_routine

from versis.advective_fluxes import calc_ZonalFlux, calc_MeridionalFlux

# in this routine, the thermodynamic time step is used instead of the dynamic one.
# this has historical reasons as with lower resolutions, the dynamics change much
# slower than the thermodynamics (thermodynamics have a daily cycle). calculating 
# the ice velocity as often as the thermodynamics was unnecessarily expensive but 
# the advection is still done with the faster thermodynamic timestep as the ice
# thickness changes inbetween dynamics timesteps.


@veros_kernel
def calc_Advection(state, field):

    '''calculate change in sea ice field due to advection'''

    # retrieve cell faces
    xA = state.variables.dyG * state.variables.iceMaskU
    yA = state.variables.dxG * state.variables.iceMaskV

    # calculate ice transport
    uTrans = state.variables.uIce * xA
    vTrans = state.variables.vIce * yA

    # make local copy of field prior to advective changes
    fieldLoc = field

    # calculate zonal advective fluxes
    ZonalFlux = calc_ZonalFlux(state, fieldLoc, uTrans)

    # update field according to zonal fluxes
    if state.settings.extensiveFld:
        fieldLoc = fieldLoc - state.settings.deltatTherm * state.variables.maskInC \
            * state.variables.recip_rA * ( npx.roll(ZonalFlux,-1,0) - ZonalFlux )
    else:
        fieldLoc = fieldLoc - state.settings.deltatTherm * state.variables.maskInC \
            * state.variables.recip_rA * state.variables.recip_hIceMean \
            * (( npx.roll(ZonalFlux,-1,1) - ZonalFlux )
            - ( npx.roll(state.variable.uTrans,-1,0) - state.variable.uTrans )
            * field)

    # calculate meridional advective fluxes
    MeridionalFlux = calc_MeridionalFlux(state, fieldLoc, vTrans)

    # update field according to meridional fluxes
    if state.settings.extensiveFld:
        fieldLoc = fieldLoc - state.settings.deltatTherm * state.variables.maskInC \
            * state.variables.recip_rA * ( npx.roll(MeridionalFlux,-1,1) - MeridionalFlux )
    else:
        fieldLoc = fieldLoc - state.settings.deltatTherm * state.variables.maskInC \
            * state.variables.recip_rA * state.variables.recip_hIceMean \
            * (( npx.roll(MeridionalFlux,-1,0) - MeridionalFlux )
            - ( npx.roll(state.variable.vTrans,-1,1) - state.variables.vTrans)
            * field)

    # apply mask
    fieldLoc = fieldLoc * state.variables.iceMask

    return fieldLoc

@veros_kernel
def do_Advections(state):

    '''retrieve changes in sea ice fields'''

    hIceMean    = calc_Advection(state, state.variables.hIceMean)
    hSnowMean   = calc_Advection(state, state.variables.hSnowMean)
    Area        = calc_Advection(state, state.variables.Area)

    return KernelOutput(hIceMean = hIceMean, hSnowMean = hSnowMean, Area = Area)

@veros_routine
def update_Advection(state):

    '''retrieve changes in sea ice fields and update state object'''

    Advection = do_Advections(state)
    state.variables.update(Advection)