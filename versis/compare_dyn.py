from veros import runtime_settings
runtime_settings.backend = 'numpy'
from veros.core.operators import numpy as npx


import matplotlib.pyplot as plt
import sys
sys.path.insert(1, "/Users/jgaertne/Documents/seaice_plugin/versis/")


from model import model


from initialize22 import state
from time import time
#state = get_state()

#model(state)
start = time()

# 1441 (2days)
print(npx.mean(state.variables.ATemp))
print('ice_start:', npx.mean(state.variables.hIceMean))

for i in range(3):
    print(i)
    model(state)
    print('ice:', npx.mean(state.variables.hIceMean))
    print('temp:', npx.mean(state.variables.TIceSnow))


end = time()

print('runtime =', end - start)

nx = 65
ny = nx
nITC = 1
olx = 2
oly = 2

# fig, axs = plt.subplots(2,2, figsize=(9, 6.5))
# ax0 = axs[0,0].pcolormesh(state.variables.uWind)
# axs[0,0].set_title('uWind')
# ax1 = axs[1,0].pcolormesh(state.variables.vWind)
# axs[1,0].set_title('vWind')
# ax2 = axs[0,1].pcolormesh(state.variables.uOcean)
# axs[0,1].set_title('uOcean')
# ax3 = axs[1,1].pcolormesh(state.variables.vOcean)
# axs[1,1].set_title('vOcean')

# plt.colorbar(ax0, ax=axs[0,0])
# plt.colorbar(ax1, ax=axs[1,0])
# plt.colorbar(ax2, ax=axs[0,1])
# plt.colorbar(ax3, ax=axs[1,1])

fig, axs = plt.subplots(2,2, figsize=(8,6))
ax0 = axs[0,0].pcolormesh(state.variables.hIceMean[oly:-oly,olx:-olx])
axs[0,0].set_title('ice thickness')
ax1 = axs[1,0].pcolormesh(state.variables.Area[oly:-oly,olx:-olx])
axs[1,0].set_title('Area')
ax2 = axs[0,1].pcolormesh(state.variables.uIce[oly:-oly,olx:-olx])
axs[0,1].set_title('uIce')
ax3 = axs[1,1].pcolormesh(state.variables.vIce[oly:-oly,olx:-olx])
axs[1,1].set_title('vIce')

plt.colorbar(ax0, ax=axs[0,0])
plt.colorbar(ax1, ax=axs[1,0])
plt.colorbar(ax2, ax=axs[0,1])
plt.colorbar(ax3, ax=axs[1,1])

fig.tight_layout()
plt.show()