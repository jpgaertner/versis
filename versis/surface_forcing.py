from veros.core.operators import numpy as npx
from veros import veros_kernel

from versis.parameters import eps_sq, eps, airTurnAngle, \
        rhoAir, airIceDrag, airIceDrag_south


@veros_kernel
def surface_forcing(state):

    '''calculate surface stress from wind and ice velocities'''

    # use turning angle (default is zero)
    sinWin = npx.sin(npx.deg2rad(airTurnAngle))
    cosWin = npx.cos(npx.deg2rad(airTurnAngle))

    ##### set up forcing fields #####

    # wind stress is computed on the center of the grid cell and
    # interpolated to u and v points later

    # calculate relative wind at c-points
    urel = state.variables.uWind - 0.5 * (
            state.variables.uIce + npx.roll(state.variables.uIce,-1,0) )
    vrel = state.variables.vWind - 0.5 * (
            state.variables.vIce + npx.roll(state.variables.vIce,-1,1) )

    # calculate wind speed and set lower boundary
    windSpeed = urel**2 + vrel**2
    windSpeed = npx.where(windSpeed < eps_sq, eps, npx.sqrt(windSpeed))

    # calculate air-ice drag coefficient
    CDAir = npx.where(state.variables.fCori < 0, airIceDrag_south, airIceDrag) * rhoAir * windSpeed
    
    # calculate surface stress
    tauX = CDAir * ( cosWin * urel - npx.sign(state.variables.fCori) * sinWin * vrel )
    tauY = CDAir * ( cosWin * vrel + npx.sign(state.variables.fCori) * sinWin * urel )

    # interpolate to u- and v-points
    tauX = 0.5 * ( tauX + npx.roll(tauX,1,0) ) * state.variables.iceMaskU
    tauY = 0.5 * ( tauY + npx.roll(tauY,1,1) ) * state.variables.iceMaskV

    return tauX, tauY