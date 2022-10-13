from veros.core.operators import numpy as npx
from veros import veros_routine, veros_kernel, KernelOutput

from versis.parameters import rhoIce, rhoSnow


@veros_kernel
def calc_AreaWS(state):

    '''calculate sea ice cover fraction centered around velocity points'''

    AreaW = 0.5 * (state.variables.Area + npx.roll(state.variables.Area,1,0))
    AreaS = 0.5 * (state.variables.Area + npx.roll(state.variables.Area,1,1))

    return KernelOutput(AreaW = AreaW, AreaS = AreaS)

@veros_routine
def update_AreaWS(state):

    '''retrieve sea ice cover fraction and update state object'''
    
    AreaWS = calc_AreaWS(state)
    state.variables.update(AreaWS)

@veros_kernel
def calc_SeaIceMass(state):

    '''calculate mass of the ice-snow system centered around c-, u-, and v-points'''

    SeaIceMassC = rhoIce * state.variables.hIceMean \
                + rhoSnow * state.variables.hSnowMean
    SeaIceMassU = 0.5 * ( SeaIceMassC + npx.roll(SeaIceMassC,1,0) )
    SeaIceMassV = 0.5 * ( SeaIceMassC + npx.roll(SeaIceMassC,1,1) )

    return KernelOutput(SeaIceMassC = SeaIceMassC,
                        SeaIceMassU = SeaIceMassU,
                        SeaIceMassV = SeaIceMassV)

@veros_routine
def update_SeaIceMass(state):
    
    '''retrieve sea ice mass and update state object'''

    SeaIceMass = calc_SeaIceMass(state)
    state.variables.update(SeaIceMass)