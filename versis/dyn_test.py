from veros import runtime_settings
runtime_settings.backend = 'jax'
from veros.core.operators import numpy as npx, update, at
from veros import veros_routine

import matplotlib.pyplot as plt
from model import model
from new_init import state


vs = state.variables

@veros_routine
def set_forcing(state):
    vs = state.variables

    vs.uWind = update(vs.uWind, at[:,:], 3)
    vs.vWind = update(vs.vWind, at[:,:], 3)

set_forcing(state)


for i in range(40):
    print(i)
    model(state)


show_forcing = False
if show_forcing:
    var = [vs.uWind[2:-2,2:-2], vs.vWind[2:-2,2:-2], vs.uOcean[2:-2,2:-2], vs.vOcean[2:-2,2:-2]]
    title = ['uWind [m/s]', 'vWind [m/s]', 'uOcean [m/s]', 'vOcean[m/s]']
else:
    var = [vs.hIceMean[2:-2,2:-2], vs.Area[2:-2,2:-2], vs.uIce[2:-2,2:-2], vs.vIce[2:-2,2:-2]]
    title = ['hIceMean [m]', 'Area [ ]', 'uIce [m/s]', 'vIce [m/s]']

fig, axs = plt.subplots(2,2, figsize=(9,6.5))
im0 = axs[0,0].pcolormesh(var[0])
axs[0,0].set_title(title[0])
plt.colorbar(im0, ax=axs[0,0])
im1 = axs[0,1].pcolormesh(var[1])
axs[0,1].set_title(title[1])
plt.colorbar(im1, ax=axs[0,1])
im2 = axs[1,0].pcolormesh(var[2])
axs[1,0].set_title(title[2])
plt.colorbar(im2, ax=axs[1,0])
im3 = axs[1,1].pcolormesh(var[3])
axs[1,1].set_title(title[3])
plt.colorbar(im3, ax=axs[1,1])

plt.show()