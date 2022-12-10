from veros import veros_routine
from veros.core.operators import numpy as npx, update, at

from versis.parameters import cpWater, rhoSea, celsius2K, gravity

@veros_routine
def copy_input(state):

    vs = state.variables

    # ocean surface velocity, temperature and salinity
    vs.uOcean = vs.u[:,:,-1,vs.tau]
    vs.vOcean = vs.v[:,:,-1,vs.tau]
    vs.theta = vs.temp[:,:,-1,vs.tau] + celsius2K
    vs.ocSalt = vs.salt[:,:,-1,vs.tau]

    # ocean surface surface total and shortwave heat flux
    vs.Qnet = - vs.forc_temp_surface * cpWater * rhoSea
    vs.Qsw = - vs.SWdown

    # sea surface height anomaly (dividing by the size of the uppermost grid cell
    # to get the right units)
    vs.ssh_an = vs.psi[:,:,vs.tau] * vs.fCori / ( gravity * vs.dzw[-1] ) 


@veros_routine
def copy_output(state):

    vs = state.variables

    # TODO all of this is useless as forc_temp_surface and surface_taux/y gets updated in
    # set_forcing_kernel at the beginning of the time step. This routine is only useful
    # if it is executed at the beginning of the time step (before set_forcing_kernel).

    # set the surface heat flux to the heat flux that is reduced due to a potential ice cover
    vs.forc_temp_surface = - vs.Qnet / ( cpWater * rhoSea )

    # all of the surface heat forcing for veros is contained in forc_temp_surface, therefore
    # no shortwave radiation has to be updated here

    # the surface salt flux vs.forc_salt_surface is set at the end of the growth routine

    # update the ocean surface stress by area weighting:
    # wind stress when there is no ice, ice-ocean stress when there is ice
    vs.surface_taux = vs.surface_taux * (1 - vs.AreaW) + vs.fu * vs.AreaW
    vs.surface_tauy = vs.surface_tauy * (1 - vs.AreaS) + vs.fv * vs.AreaS