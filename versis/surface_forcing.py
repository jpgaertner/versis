from veros.core.operators import numpy as npx
from veros import veros_kernel


@veros_kernel
def surface_forcing(state):

    '''calculate surface stress from wind and ice velocities'''

    vs = state.variables
    sett = state.settings

    # use turning angle (default is zero)
    sinWin = npx.sin(npx.deg2rad(sett.airTurnAngle))
    cosWin = npx.cos(npx.deg2rad(sett.airTurnAngle))

    ##### set up forcing fields #####

    # wind stress is computed on the center of the grid cell and
    # interpolated to u and v points later

    # calculate relative wind at c-points
    urel = vs.uWind - 0.5 * ( vs.uIce + npx.roll(vs.uIce,-1,0) )
    vrel = vs.vWind - 0.5 * ( vs.vIce + npx.roll(vs.vIce,-1,1) )

    # calculate wind speed and set lower boundary
    windSpeed_sq = urel**2 + vrel**2
    windSpeed = npx.where(windSpeed_sq < sett.wSpeedMin**2, sett.wSpeedMin, npx.sqrt(windSpeed_sq))

    # calculate air-ice drag coefficient
    CDAir = npx.where(vs.fCori < 0, sett.airIceDrag_south, sett.airIceDrag) * sett.rhoAir * windSpeed
    
    # calculate surface stress
    tauX = CDAir * ( cosWin * urel - npx.sign(vs.fCori) * sinWin * vrel )
    tauY = CDAir * ( cosWin * vrel + npx.sign(vs.fCori) * sinWin * urel )

    # interpolate to u- and v-points
    tauX = 0.5 * ( tauX + npx.roll(tauX,1,0) ) * vs.iceMaskU
    tauY = 0.5 * ( tauY + npx.roll(tauY,1,1) ) * vs.iceMaskV

    return tauX, tauY