from veros.core.operators import numpy as npx
from veros import veros_kernel, KernelOutput, veros_routine

from versis.parameters import *


@veros_kernel
def calc_IceStrength(state):

    '''calculate ice strength (= maximum compressive stress)
    from ice thickness and ice cover fraction
    '''

    SeaIceStrength = pStar * state.variables.hIceMean \
                * npx.exp(-cStar * (1 - state.variables.Area)) * state.variables.iceMask
    return KernelOutput(SeaIceStrength = SeaIceStrength)

@veros_routine
def update_IceStrength(state):

    '''retrieve sea ice strength and update state object'''

    IceStrength = calc_IceStrength(state)
    state.variables.update(IceStrength)

@veros_kernel
def ocean_drag_coeffs(state,uIce,vIce):

    '''calculate linear ice-water drag coefficient from ice and ocean velocities
    (this coefficient creates a linear relationship between
    ice-ocean stress difference and ice-ocean velocity difference)
    '''

    # get ice-water drag coefficient times density
    dragCoeff = npx.where(state.variables.fCori < 0, waterIceDrag_south, waterIceDrag) * rhoSea

    # calculate component-wise velocity differences at velocity points
    du = (uIce - state.variables.uOcean)*state.variables.maskInU
    dv = (vIce - state.variables.vOcean)*state.variables.maskInV

    # calculate velocity difference at c-point
    tmpVar = 0.25 * ( (du + npx.roll(du,-1,0))**2
                    +  (dv + npx.roll(dv,-1,1))**2 )

    # calculate linear drag coefficient and apply mask
    cDrag = npx.where(dragCoeff**2 * tmpVar > cDragMin**2,
                        dragCoeff * npx.sqrt(tmpVar), cDragMin)
    cDrag = cDrag * state.variables.iceMask

    return cDrag

@veros_kernel
def basal_drag_coeffs(state, uIce, vIce):

    '''calculate basal drag coefficient to account for the formation of 
    landfast ice in shallow waters due to the formation of ice keels
    (Lemieux et al., 2015)
    '''

    # absolute value of the ice velocity at c-points
    tmpFld = 0.25 * ( (uIce*state.variables.maskInU)**2
                    + npx.roll(uIce*state.variables.maskInU,-1,0)**2
                    + (vIce*state.variables.maskInV)**2
                    + npx.roll(vIce*state.variables.maskInV,-1,1)**2 )

    # include velocity parameter U0 to avoid singularities
    tmpFld = basalDragK2 / npx.sqrt(tmpFld + basalDragU0**2)

    # critical ice height that allows for the formation of landfast ice
    hCrit = npx.abs(state.variables.R_low) * state.variables.Area / basalDragK1

    # soft maximum for better differentiability:
    # max(a,b;k) = ln(exp(k*a)+exp(k*b))/k
    # In our case, b=0, so exp(k*b) = 1.
    # max(a,0;k) = ln(exp(k*a)+1)/k
    # If k*a gets too large, EXP will overflow, but for the anticipated
    # values of hActual < 100m, and k=10, this should be very unlikely
    fac = 10. 
    recip_fac = 1. / fac
    cBot = npx.where(state.variables.Area > 0.01,
            tmpFld
                * npx.log(npx.exp(fac * (state.variables.hIceMean-hCrit)) + 1.)
                * recip_fac * npx.exp(-cBasalStar * (1. - state.variables.Area)),
                0.)

    return cBot

@veros_kernel
def strainrates(state, uIce, vIce):

    '''calculate strain rate tensor components from ice velocities'''

    # some abbreviations at c-points
    dudx = ( npx.roll(uIce,-1,axis=0) - uIce ) * state.variables.recip_dxU
    uave = ( npx.roll(uIce,-1,axis=0) + uIce ) * 0.5
    dvdy = ( npx.roll(vIce,-1,axis=1) - vIce ) * state.variables.recip_dyV
    vave = ( npx.roll(vIce,-1,axis=1) + vIce ) * 0.5

    # calculate strain rates at c-points
    e11 = ( dudx + vave * state.variables.k2AtC ) * state.variables.maskInC
    e22 = ( dvdy + uave * state.variables.k1AtC ) * state.variables.maskInC

    # some abbreviations at z-points
    dudy = ( uIce - npx.roll(uIce,1,axis=1) ) * state.variables.recip_dyU
    uave = ( uIce + npx.roll(uIce,1,axis=1) ) * 0.5
    dvdx = ( vIce - npx.roll(vIce,1,axis=0) ) * state.variables.recip_dxV
    vave = ( vIce + npx.roll(vIce,1,axis=0) ) * 0.5

    # calculate strain rate at z-points
    mskZ = state.variables.iceMask*npx.roll(state.variables.iceMask,1,axis=0)
    mskZ =    mskZ*npx.roll(   mskZ,1,axis=1)
    e12 = 0.5 * ( dudy + dvdx - state.variables.k1AtZ * vave - state.variables.k2AtZ * uave ) * mskZ
    if state.settings.noSlip:
        hFacU = state.variables.iceMaskU - npx.roll(state.variables.iceMaskU,1,axis=1)
        hFacV = state.variables.iceMaskV - npx.roll(state.variables.iceMaskV,1,axis=0)
        e12   = e12 + ( 2.0 * uave * state.variables.recip_dyU * hFacU
                      + 2.0 * vave * state.variables.recip_dxV * hFacV )

    if state.settings.noSlip and state.settings.secondOrderBC:
        hFacU = ( state.variables.iceMaskU - npx.roll(state.variables.iceMaskU,1,1) ) / 3.
        hFacV = ( state.variables.iceMaskV - npx.roll(state.variables.iceMaskV,1,0) ) / 3.
        hFacU = hFacU * (npx.roll(state.variables.iceMaskU, 2,1) * npx.roll(state.variables.iceMaskU,1,1)
                       + npx.roll(state.variables.iceMaskU,-1,1) * state.variables.iceMaskU )
        hFacV = hFacV * (npx.roll(state.variables.iceMaskV, 2,0) * npx.roll(state.variables.iceMaskV,1,0)
                       + npx.roll(state.variables.iceMaskV,-1,0) * state.variables.iceMaskV )
        # right hand sided dv/dx = (9*v(i,j)-v(i+1,j))/(4*dxv(i,j)-dxv(i+1,j))
        # according to a Taylor expansion to 2nd order. We assume that dxv
        # varies very slowly, so that the denominator simplifies to 3*dxv(i,j),
        # then dv/dx = (6*v(i,j)+3*v(i,j)-v(i+1,j))/(3*dxv(i,j))
        #            = 2*v(i,j)/dxv(i,j) + (3*v(i,j)-v(i+1,j))/(3*dxv(i,j))
        # the left hand sided dv/dx is analogously
        #            = - 2*v(i-1,j)/dxv(i,j)-(3*v(i-1,j)-v(i-2,j))/(3*dxv(i,j))
        # the first term is the first order part, which is already added.
        # For e12 we only need 0.5 of this gradient and vave = is either
        # 0.5*v(i,j) or 0.5*v(i-1,j) near the boundary so that we need an
        # extra factor of 2. This explains the six. du/dy is analogous.
        # The masking is ugly, but hopefully effective.
        e12 = e12 + 0.5 * (
            state.variables.recip_dyU * ( 6. * uave
                          - npx.roll(uIce, 2,1) * npx.roll(state.variables.iceMaskU,1,1)
                          - npx.roll(uIce,-1,1) * state.variables.iceMaskU ) * hFacU
          + state.variables.recip_dxV * ( 6. * vave
                          - npx.roll(vIce, 2,0) * npx.roll(state.variables.iceMaskV,1,0)
                          - npx.roll(vIce,-1,0) * state.variables.iceMaskV ) * hFacV
        )

    return e11, e22, e12

@veros_kernel
def viscosities(state, e11,e22,e12):

    """calculate bulk viscosity zeta, shear viscosity eta, and ice pressure
    from strain rate tensor components and ice strength.
    if pressReplFac = 1, a replacement pressure is used to avoid
    stresses without velocities (Hibler and Ib, 1995).
    with tensileStrFac != 1, a resistance to tensile stresses can be included
    (König Beatty and Holland, 2010).
    """

    recip_PlasDefCoeffSq = 1. / PlasDefCoeff**2

    # interpolate squares of e12 to c-points after weighting them with the
    # area centered around z-points
    e12Csq = state.variables.rAz * e12**2
    e12Csq =                     e12Csq + npx.roll(e12Csq,-1,1)
    e12Csq = 0.25 * state.variables.recip_rA * ( e12Csq + npx.roll(e12Csq,-1,0) )

    # calculate Delta from the normal strain rate (e11+e22)
    # and the shear strain rate sqrt( (e11-e22)**2 + 4 * e12**2) )
    deltaSq = (e11+e22)**2 + recip_PlasDefCoeffSq * (
        (e11-e22)**2 + 4. * e12Csq )
    deltaC = npx.sqrt(deltaSq)

    # use regularization to avoid singularies of zeta
    # TODO implement smooth regularization after comparing with the MITgcm
    # smooth regularization of delta for better differentiability
    # deltaCreg = deltaC + deltaMin
    # deltaCreg = npx.sqrt( deltaSq + deltaMin**2 )
    deltaCreg = npx.maximum(deltaC,deltaMin)

    # calculate viscosities
    zeta = 0.5 * ( state.variables.SeaIceStrength * (1 + tensileStrFac) ) / deltaCreg
    eta  = zeta * recip_PlasDefCoeffSq

    # calculate ice pressure
    press = 0.5 * ( state.variables.SeaIceStrength * (1 - pressReplFac)
              + 2. * zeta * deltaC * pressReplFac / (1 + tensileStrFac)
             ) * (1 - tensileStrFac)

    return zeta, eta, press

@veros_kernel
def stress(state,e11, e22, e12, zeta, eta, press):

    '''calculate stress tensor components'''

    from versis.averaging import c_point_to_z_point
    sig11 = zeta*(e11 + e22) + eta*(e11 - e22) - press
    sig22 = zeta*(e11 + e22) - eta*(e11 - e22) - press
    sig12 = 2. * e12 * c_point_to_z_point(state,eta)

    return sig11, sig22, sig12

@veros_kernel
def stressdiv(state,sig11, sig22, sig12):

    '''calculate divergence of stress tensor'''

    stressDivX = (
          sig11*state.variables.dyV - npx.roll(sig11*state.variables.dyV, 1,axis=0)
        - sig12*state.variables.dxV + npx.roll(sig12*state.variables.dxV,-1,axis=1)
    ) * state.variables.recip_rAu
    stressDivY = (
          sig22*state.variables.dxU - npx.roll(sig22*state.variables.dxU, 1,axis=1)
        - sig12*state.variables.dyU + npx.roll(sig12*state.variables.dyU,-1,axis=0)
    ) * state.variables.recip_rAv

    return stressDivX, stressDivY