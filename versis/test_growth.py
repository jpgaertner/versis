from veros import runtime_settings
runtime_settings.backend = 'numpy'
from veros.core.operators import numpy as npx
from veros import veros_routine
import matplotlib.pyplot as plt
from growth import calc_growth, update_Growth
from test_init import state

vs = state.variables
sett = state.settings

timesteps = 30

ones2d = npx.ones_like(vs.maskInC)
ones3d = npx.ones_like((*vs.maskInC,1))

@veros_routine
def forcing(state):
    vs = state.variables

    vs.Qnet = ones2d * 252.19888563808655
    vs.Qsw = ones2d * 0
    vs.snowfall = ones2d * 0
    vs.precip = ones2d * 0
    vs.evap = ones2d * 0
    vs.LWdown = ones2d * 80
    vs.SWdown = ones2d * 0
    vs.ATemp = ones2d * 243
    vs.wSpeed = ones2d * 2
    vs.aqh = ones2d * 0
    vs.theta = ones2d * sett.celsius2K - 1.9
    vs.ocSalt = ones2d * 29

@veros_routine
def update(state):
    vs = state.variables

    vs.hIceMean = hIceMean
    vs.hSnowMean = hSnowMean
    vs.Area = Area
    vs.TSurf = TSurf

forcing(state)

ice = npx.array([vs.hIceMean[0,0]])
snow = npx.array([vs.hSnowMean[0,0]])
iceTemp = npx.array([vs.TSurf[0,0]])
area = npx.array([vs.Area[0,0]])
qnet_out = npx.array([0])
empmr = npx.array([])


for i in range(timesteps):
    # hIceMean, hSnowMean, Area, TSurf, saltflux, EmPmR, forc_salt_surface, Qsw, Qnet, \
    #     SeaIceLoad, IcePenetSW = calc_growth(state, vs)

    # update(state)

    # ice = npx.append(ice, hIceMean[0,0])
    # snow = npx.append(snow, hSnowMean[0,0])
    # iceTemp = npx.append(iceTemp, TSurf[0,0])
    # area = npx.append(area, Area[0,0])
    # qnet_out = npx.append(qnet_out, Qnet[0,0])
    # empmr = npx.append(empmr, EmPmR[0,0])

    update_Growth(state)

    ice = npx.append(ice, state.variables.hIceMean[0,0])
    snow = npx.append(snow, state.variables.hSnowMean[0,0])
    iceTemp = npx.append(iceTemp, state.variables.TSurf[0,0])
    area = npx.append(area, state.variables.Area[0,0])

fig, axs = plt.subplots(2,2, figsize=(10,6))
axs[0,0].plot(ice)
axs[0,0].set_ylabel("ice thickness [m]")
axs[0,1].plot(area)
axs[0,1].set_ylabel("ice concentration")
axs[1,0].plot(qnet_out)
axs[1,0].set_ylabel("qnet after ice")
axs[1,1].plot(empmr)
axs[1,1].set_ylabel("empmr")

fig.tight_layout()
plt.show()