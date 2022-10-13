from veros import veros_routine
from veros.core.operators import numpy as npx, update, at

from versis.parameters import heatCapacity, rhoSea, celsius2K, gravity

@veros_routine
def copy_input(state):
    vs = state.variables

    # ocean surface velocity, temperature and salinity
    vs.uOcean = vs.u[:,:,-1,vs.tau]
    vs.vOcean = vs.v[:,:,-1,vs.tau]
    vs.theta = vs.temp[:,:,-1,vs.tau] + celsius2K
    vs.ocSalt = vs.salt[:,:,-1,vs.tau]

    # ocean depth (topography)
    vs.R_low = vs.ht

    # ocean surface surface total and shortwave heat flux
    vs.Qnet = - vs.forc_temp_surface * heatCapacity * rhoSea
    vs.Qsw = - vs.SWDown

    # sea surface height anomaly
    vs.ssh_an = vs.psi * vs.fCori / gravity


@veros_routine
def copy_output(state):
    vs = state.variables

    # set the surface heat flux to the heat flux that is reduced due to a potential ice cover
    vs.forc_temp_surface = - vs.Qnet / ( heatCapacity * rhoSea )

    # update the ocean surface stress by area weighting:
    # wind stress when there is no ice, ice-ocean stress when there is ice
    vs.surface_taux = state.variables.surface_taux * (1 - state.variables.AreaW) \
                    + state.variables.fu * state.variables.AreaW
    vs.surface_tauy = state.variables.surface_tauy * (1 - state.variables.AreaS) \
                    + state.variables.fv * state.variables.AreaS