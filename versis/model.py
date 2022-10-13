from veros import veros_routine

from versis.clean_up import update_clean_up_advection, update_ridging
from versis.dynamics_routines import update_IceStrength
from versis.advection import update_Advection
from versis.dynsolver import update_IceVelocities, update_SurfaceForcing
from versis.ocean_stress import update_OceanStress
from versis.growth import update_Growth
from versis.area_mass import update_AreaWS, update_SeaIceMass
from versis.copy_veros_variables import copy_input, copy_output


@veros_routine
def model(state):

    # copy veros variables onto local ones
    copy_input(state)

    # calculate sea ice mass centered around c, u, v points
    update_SeaIceMass(state)

    # calculate surface forcing due to wind
    update_SurfaceForcing(state)

    # calculate ice strength
    update_IceStrength(state)

    # calculate sea ice cover fraction centered around u,v points
    update_AreaWS(state)

    # calculate ice velocities
    update_IceVelocities(state)

    # calculate stresses on ocean surface
    update_OceanStress(state)

    # calculate change in sea ice fields due to advection
    update_Advection(state)

    # correct overshoots and other pathological cases after advection
    update_clean_up_advection(state)

    # cut off ice cover fraction at 1 after advection
    update_ridging(state)

    # calculate thermodynamic ice growth
    update_Growth(state)

    # copy local variables onto veros
    # copy_output(state)