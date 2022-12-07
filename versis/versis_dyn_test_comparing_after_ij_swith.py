from veros import runtime_settings
runtime_settings.backend = 'jax'
from veros.core.operators import numpy as npx, update, at
from veros import veros_routine
import matplotlib.pyplot as plt
from time import time

from fill_overlap import fill_overlap
from model import model
from gendata import uo,vo,uwind,vwind
from new_init import state


v_path = "/Users/jgaertne/Documents/seaice_plugin/versis"

v_data = npx.load(v_path + '/vfile.npy')
v_ice = v_data[0]
v_area = v_data[1]
v_u = v_data[2]
v_v = v_data[3]
v_uWind = v_data[4]
v_vWind = v_data[5]
v_uOcean = v_data[6]
v_vOcean = v_data[7]

olx = 2
oly = 2

@veros_routine
def set_forcing(state):
    vs = state.variables

    vs.uOcean = update(vs.uOcean, at[olx:-olx,oly:-oly], uo.T)
    vs.uOcean = fill_overlap(state,vs.uOcean)
    vs.vOcean = update(vs.vOcean, at[olx:-olx,oly:-oly], vo.T)
    vs.vOcean = fill_overlap(state,vs.vOcean)
    vs.uWind = update(vs.uWind, at[olx:-olx,oly:-oly], uwind[0,:,:].T)
    vs.uWind = fill_overlap(state,vs.uWind)
    vs.vWind = update(vs.vWind, at[olx:-olx,oly:-oly], vwind[0,:,:].T)
    vs.vWind = fill_overlap(state,vs.vWind)

    # vs.uWind = update(vs.uWind, at[:,:], 10)
    # vs.vWind = update(vs.vWind, at[:,:], 10)


set_forcing(state)

start = time()

for i in range(100):
    print(i)
    model(state)


end = time()

print('runtime =', end - start)


olx = 2
oly = 2

fig, axs = plt.subplots(2,2, figsize=(9, 6.5))
ax0 = axs[0,0].pcolormesh((state.variables.hIceMean.T -
                            v_ice) [oly:-oly-1,olx:-olx-1])
axs[0,0].set_title('ice anomaly')
ax1 = axs[1,0].pcolormesh((state.variables.Area.T -
                            v_area) [oly:-oly-1,olx:-olx-1])
axs[1,0].set_title('area anomaly')
ax2 = axs[0,1].pcolormesh((state.variables.uIce.T -
                            v_u) [oly:-oly-1,olx:-olx-1])
axs[0,1].set_title('uIce anomaly')
ax3 = axs[1,1].pcolormesh((state.variables.vIce.T -
                            v_v) [oly:-oly-1,olx:-olx-1])
axs[1,1].set_title('vIce anomaly')

# fig, axs = plt.subplots(2,2, figsize=(8,6))
# ax0 = axs[0,0].pcolormesh(state.variables.hIceMean.T[oly:-oly-1,olx:-olx-1])
# axs[0,0].set_title('ice thickness')
# ax1 = axs[1,0].pcolormesh(state.variables.Area.T[oly:-oly-1,olx:-olx-1])
# axs[1,0].set_title('Area')
# ax2 = axs[0,1].pcolormesh(state.variables.uIce.T[oly:-oly-1,olx:-olx-1])
# axs[0,1].set_title('uIce')
# ax3 = axs[1,1].pcolormesh(state.variables.vIce.T[oly:-oly-1,olx:-olx-1])
# axs[1,1].set_title('vIce')

plt.colorbar(ax0, ax=axs[0,0])
plt.colorbar(ax1, ax=axs[1,0])
plt.colorbar(ax2, ax=axs[0,1])
plt.colorbar(ax3, ax=axs[1,1])

fig.tight_layout()
plt.show()