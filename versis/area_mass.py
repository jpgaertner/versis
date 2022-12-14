from veros.core.operators import numpy as npx
from veros import veros_routine, veros_kernel, KernelOutput


@veros_kernel
def calc_AreaWS(state):

    '''calculate sea ice cover fraction centered around velocity points'''

    vs = state.variables

    AreaW = 0.5 * (vs.Area + npx.roll(vs.Area,1,0))
    AreaS = 0.5 * (vs.Area + npx.roll(vs.Area,1,1))

    return KernelOutput(AreaW = AreaW, AreaS = AreaS)

@veros_routine
def update_AreaWS(state):

    '''retrieve sea ice cover fraction and update state object'''
    
    AreaWS = calc_AreaWS(state)
    state.variables.update(AreaWS)

@veros_kernel
def calc_SeaIceMass(state):

    '''calculate mass of the ice-snow system centered around c-, u-, and v-points'''

    vs = state.variables
    sett = state.settings

    SeaIceMassC = sett.rhoIce * vs.hIceMean \
                + sett.rhoSnow * vs.hSnowMean
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