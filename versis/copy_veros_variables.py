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
    vs.Qsw = - vs.SWdown

    # sea surface height anomaly (dividing by the size of the uppermost grid cell
    # to get the right units)
    vs.ssh_an = vs.psi[:,:,vs.tau] * vs.fCori / ( gravity * vs.dzw[-1] ) 


@veros_routine
def copy_output(state):

    vs = state.variables

    #TODO this is useless as forc_temp_surface gets update in set_forcing_kernel at the beginning
    # of the time step. This is only useful if versis is executed at the beginning of the time step.
    # then add also something to update the shortwave flux arriving at the ocean surface
    # set the surface heat flux to the heat flux that is reduced due to a potential ice cover
    vs.forc_temp_surface = - vs.Qnet / ( heatCapacity * rhoSea )


    # the surface salt flux vs.forc_salt_surface is set at the end of the growth routine

    # update the ocean surface stress by area weighting:
    # wind stress when there is no ice, ice-ocean stress when there is ice
    vs.surface_taux = vs.surface_taux * (1 - vs.AreaW) + vs.fu * vs.AreaW
    vs.surface_tauy = vs.surface_tauy * (1 - vs.AreaS) + vs.fv * vs.AreaS