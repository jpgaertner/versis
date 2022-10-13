from versis.model import model
from versis.set_inits import set_inits
from versis.variables import VARIABLES
from versis.settings import SETTINGS


__VEROS_INTERFACE__ = dict(
    name = 'versis',
    setup_entrypoint = set_inits,
    run_entrypoint = model,
    settings = SETTINGS,
    variables = VARIABLES,
)