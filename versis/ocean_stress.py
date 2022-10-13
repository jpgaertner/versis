from veros.core.operators import numpy as npx
from veros import veros_kernel, veros_routine, KernelOutput

from versis.parameters import waterTurnAngle
from versis.fill_overlap import fill_overlap_uv
from versis.dynamics_routines import ocean_drag_coeffs


@veros_kernel
def calc_OceanStress(state):

    '''calculate stresses on ocean surface from ocean and ice velocities'''

    # get linear drag coefficient at c-point
    cDrag = ocean_drag_coeffs(state, state.variables.uIce, state.variables.vIce)

    # use turning angle (default is zero)
    sinWat = npx.sin(npx.deg2rad(waterTurnAngle))
    cosWat = npx.cos(npx.deg2rad(waterTurnAngle))

    # calculate component-wise velocity difference of ice and ocean surface
    du = state.variables.uIce - state.variables.uOcean
    dv = state.variables.vIce - state.variables.vOcean

    # interpolate to c-points
    duAtC = 0.5 * (du + npx.roll(du,-1,0))
    dvAtC = 0.5 * (dv + npx.roll(dv,-1,1))

    # calculate forcing on ocean surface in u- and v-direction
    fuLoc = 0.5 * (cDrag + npx.roll(cDrag,1,0)) * cosWat * du \
        - npx.sign(state.variables.fCori) * sinWat * 0.5 * (
            cDrag * dvAtC + npx.roll(cDrag * dvAtC,1,1) )
    fvLoc = 0.5 * (cDrag + npx.roll(cDrag,1,1)) * cosWat * dv \
        + npx.sign(state.variables.fCori) * sinWat * 0.5 * (
            cDrag * duAtC + npx.roll(cDrag * duAtC,1,0) )

    # update forcing for ice covered area
    fu = (1 - state.variables.AreaW) * state.variables.fu + state.variables.AreaW * fuLoc
    fv = (1 - state.variables.AreaS) * state.variables.fv + state.variables.AreaS * fvLoc

    # fill overlaps
    fu, fv = fill_overlap_uv(state,fu,fv)

    return KernelOutput(fu = fu, fv = fv)

@veros_routine
def update_OceanStress(state):

    '''retrieve stresses on ocean surface and update state object'''

    OceanStress = calc_OceanStress(state)
    state.variables.update(OceanStress)