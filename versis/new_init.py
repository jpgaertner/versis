from veros.state import VerosState

from versis.settings import SETTINGS
from versis.variables import VARIABLES
from versis.set_inits import set_inits

nx = 65
ny = nx
olx = 2
oly = 2
nITC = 1

dimensions = dict(x=nx+2*olx, y=ny+2*oly, z=nITC)

state = VerosState(VARIABLES, SETTINGS, dimensions)
state.initialize_variables()
set_inits(state)
