from veros.core.operators import numpy as npx
from veros.core.operators import update, at
from veros import veros_kernel, KernelOutput, veros_routine

from versis.parameters import si_eps, celsius2K, area_floor


@veros_kernel
def clean_up_advection(state):

    '''clean up overshoots and other pathological cases after advection'''

    # case 1: negative values
    # calculate overshoots of ice and snow thickness
    os_hIceMean = npx.maximum(-state.variables.hIceMean, 0)
    os_hSnowMean = npx.maximum(-state.variables.hSnowMean, 0)

    # cut off thicknesses and area at zero
    hIceMean = npx.maximum(state.variables.hIceMean, 0)
    hSnowMean = npx.maximum(state.variables.hSnowMean, 0)
    Area = npx.maximum(state.variables.Area, 0)

    # case 2: very thin ice
    # set thicknesses to zero if the ice thickness is very small
    thinIce = (hIceMean <= si_eps)
    hIceMean *= ~thinIce
    hSnowMean *= ~thinIce
    TSurf = npx.where(thinIce, celsius2K, state.variables.TSurf)

    # case 3: area but no ice and snow
    # set area to zero if no ice or snow is present
    Area = npx.where((hIceMean == 0) & (hSnowMean == 0), 0, Area)

    # case 4: very small area
    # introduce lower boundary for the area (if ice or snow is present)
    Area = npx.where((hIceMean > 0) | (hSnowMean > 0),
                        npx.maximum(Area, area_floor), Area)

    return KernelOutput(hIceMean = hIceMean,
                        hSnowMean = hSnowMean,
                        Area = Area,
                        TSurf = TSurf,
                        os_hIceMean = os_hIceMean,
                        os_hSnowMean = os_hSnowMean)

@veros_routine
def update_clean_up_advection(state):

    '''retrieve clean up and update state object'''

    CleanUpAdvection = clean_up_advection(state)
    state.variables.update(CleanUpAdvection)

@veros_kernel
def ridging(state):

    '''cut off ice cover fraction at 1 after advection to account for ridging'''
    Area = npx.minimum(state.variables.Area, 1)

    return KernelOutput(Area = Area)

@veros_routine
def update_ridging(state):

    '''retrieve ice cover fraction cutoff and update state object'''
    
    Ridging = ridging(state)
    state.variables.update(Ridging)