from veros import veros_routine
from veros.core.operators import numpy as npx
from veros.core.operators import update, at

from versis.fill_overlap import fill_overlap
from versis.parameters import celsius2K, rhoIce, rhoSnow


@veros_routine
def set_inits(state):
    vs = state.variables
    st = state.settings

    ones2d = npx.ones_like(vs.maskInC)

    vs.TSurf = ones2d * 273

    # if veros is used, all other variables are either copied in copy_veros_variables,
    # set in global_4deg (set_forcing_kernel), or have 0 as initial value
    useVeros = True
    if not useVeros:
        vs.dxC = ones2d * st.gridcellWidth
        vs.dyC = vs.dxC
        vs.dxG = vs.dxC
        vs.dyG = vs.dxC
        vs.dxU = vs.dxC
        vs.dyU = vs.dxC
        vs.dxV = vs.dxC
        vs.dyV = vs.dxC

        vs.recip_dxC = 1 / vs.dxC
        vs.recip_dyC = 1 / vs.dyC
        vs.recip_dxG = 1 / vs.dxG
        vs.recip_dyG = 1 / vs.dyG
        vs.recip_dxU = 1 / vs.dxU
        vs.recip_dyU = 1 / vs.dyU
        vs.recip_dxV = 1 / vs.dxV
        vs.recip_dyV = 1 / vs.dyV

        vs.rA = vs.dxU * vs.dyV
        vs.rAz = vs.dxV * vs.dyU
        vs.rAu = vs.dxC * vs.dyG
        vs.rAv = vs.dxG * vs.dyC

        vs.recip_rA = 1 / vs.rA
        vs.recip_rAz = 1 / vs.rAz
        vs.recip_rAu = 1 / vs.rAu
        vs.recip_rAv = 1 / vs.rAv

        vs.maskInC = ones2d
        vs.maskInC = update(vs.maskInC, at[-st.olx-1,:], 0)
        vs.maskInC = update(vs.maskInC, at[:,-st.oly-1], 0)
        vs.maskInC = fill_overlap(state,vs.maskInC)
        vs.maskInU = vs.maskInC * npx.roll(vs.maskInC,1,axis=0)
        vs.maskInU = fill_overlap(state,vs.maskInU)
        vs.maskInV = vs.maskInC * npx.roll(vs.maskInC,1,axis=1)
        vs.maskInV = fill_overlap(state,vs.maskInV)

        vs.iceMask = vs.maskInC
        vs.iceMaskU = vs.maskInU
        vs.iceMaskV = vs.maskInV

        vs.hIceMean = ones2d * 1
        vs.hSnowMean = ones2d * 1
        vs.Area = ones2d * 1
        state.variables.SeaIceLoad  = ones2d * (rhoIce * state.variables.hIceMean
                                                + rhoSnow * state.variables.hSnowMean)
        # vs.TIceSnow = npx.ones((*vs.iceMask.shape,st.nITC)) * 273
        vs.TSurf = ones2d * 273
        vs.wSpeed = ones2d * 2
        vs.ocSalt = ones2d * 29
        vs.theta = ones2d * celsius2K - 1.9
        vs.Qnet = ones2d * 252.19888563808655
        vs.LWdown = ones2d * 80
        vs.ATemp = ones2d * 243