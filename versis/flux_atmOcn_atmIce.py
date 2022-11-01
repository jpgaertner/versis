import os
import netCDF4
import veros.tools
from veros.core.operators import numpy as npx
from veros.core.operators import update, at
import matplotlib.pyplot as plt

import versis.constants as ct
from versis.utilities import *


def flux_atmIce(mask, rbot, zbot, ubot, vbot, qbot, tbot, thbot, ts):
    """atm/ice fluxes calculation

    Arguments:
        mask (:obj:`ndarray`): ocn domain mask       0 <=> out of domain
        rbot (:obj:`ndarray`): atm density           (Pa)
        zbot (:obj:`ndarray`): atm level height      (m)
        ubot (:obj:`ndarray`): atm u wind            (m/s)
        vbot (:obj:`ndarray`): atm v wind            (m/s)
        qbot (:obj:`ndarray`): atm specific humidity (kg/kg)
        tbot (:obj:`ndarray`): atm T                 (K)
        thbot(:obj:`ndarray`): atm potential T       (K)
        ts   (:obj:`ndarray`): ocn temperature       (K)

    Returns:
        sen  (:obj:`ndarray`): heat flux: sensible    (W/m^2)
        lat  (:obj:`ndarray`): heat flux: latent      (W/m^2)
        lwup (:obj:`ndarray`): heat flux: lw upward   (W/m^2)
        evap (:obj:`ndarray`): water flux: evap  ((kg/s)/m^2)
        taux (:obj:`ndarray`): surface stress, zonal      (N)
        tauy (:obj:`ndarray`): surface stress, maridional (N)
        tref (:obj:`ndarray`): diag:  2m ref height T     (K)
        qref (:obj:`ndarray`): diag:  2m ref humidity (kg/kg)

    Reference:
        - Large, W. G., & Pond, S. (1981). Open Ocean Momentum Flux Measurements in Moderate to Strong Winds,
        Journal of Physical Oceanography, 11(3), pp. 324-336
        - Large, W. G., & Pond, S. (1982). Sensible and Latent Heat Flux Measurements over the Ocean,
        Journal of Physical Oceanography, 12(5), 464-482.
        - https://svn-ccsm-release.cgd.ucar.edu/model_versions/cesm1_0_5/models/csm_share/shr/shr_flux_mod.F90
    """

    vmag = npx.maximum(ct.UMIN_I, npx.sqrt((ubot[...])**2 + (vbot[...])**2))

    # virtual potential temperature (K)
    thvbot = thbot[...] * (1.0 + ct.ZVIR * qbot[...])

    # sea surface humidity (kg/kg)
    ssq = qsat(ts[...]) / rbot[...]

    # potential temperature diff. (K)
    delt = thbot[...] - ts[...]

    # specific humidity diff (kg/kg)
    delq = qbot[...] - ssq[...]

    alz = npx.log(zbot[...] / ct.ZREF)
    cp = ct.CPDAIR * (1.0 + ct.CPVIR * ssq[...])
    ct.LTHEAT = ct.LATICE + ct.LATVAP

    # First estimate of Z/L and ustar, tstar and qstar

    # neutral coefficients, z/L = 0.0
    rdn = ct.KARMAN / npx.log(ct.ZREF / ct.ZZSICE)
    rhn = rdn
    ren = rdn

    ustar = rdn * vmag[...]
    tstar = rhn * delt[...]
    qstar = ren * delq[...]

    # compute stability & evaluate all stability functions
    hol = ct.KARMAN * ct.G * zbot[...] *\
        (tstar[...] / thvbot[...] + qstar[...]
         / (1.0 / ct.ZVIR + qbot[...])) / ustar[...]**2
    hol[...] = npx.minimum(npx.abs(hol[...]), 10.0) * npx.sign(hol[...])
    stable = 0.5 + 0.5 * npx.sign(hol[...])
    xsq = npx.maximum(npx.sqrt(npx.abs(1.0 - 16.0 * hol[...])), 1.0)
    xqq = npx.sqrt(xsq[...])
    psimh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq[...])
    psixh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])

    # shift all coeffs to measurement height and stability
    rd = rdn / (1.0 + rdn / ct.KARMAN * (alz[...] - psimh[...]))
    rh = rhn / (1.0 + rhn / ct.KARMAN * (alz[...] - psixh[...]))
    re = ren / (1.0 + ren / ct.KARMAN * (alz[...] - psixh[...]))

    # update ustar, tstar, qstar using updated, shifted coeffs
    ustar = rd[...] * vmag[...]
    tstar = rh[...] * delt[...]
    qstar = re[...] * delq[...]

    # Iterate to converge on Z/L, ustar, tstar and qstar

    # compute stability & evaluate all stability functions
    hol = ct.KARMAN * ct.G * zbot[...] *\
        (tstar[...] / thvbot[...] + qstar[...]
         / (1.0 / ct.ZVIR + qbot[...])) / ustar[...]**2
    hol[...] = npx.minimum(npx.abs(hol[...]), 10.0) * npx.sign(hol[...])
    stable[...] = 0.5 + 0.5 * npx.sign(hol[...])
    xsq[...] = npx.maximum(npx.sqrt(npx.abs(1.0 - 16.0 * hol[...])), 1.0)
    xqq[...] = npx.sqrt(xsq[...])
    psimh[...] = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq[...])
    psixh[...] = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])

    # shift all coeffs to measurement height and stability
    rd[...] = rdn / (1.0 + rdn / ct.KARMAN * (alz[...] - psimh[...]))
    rh[...] = rhn / (1.0 + rhn / ct.KARMAN * (alz[...] - psixh[...]))
    re[...] = ren / (1.0 + ren / ct.KARMAN * (alz[...] - psixh[...]))

    # update ustar, tstar, qstar using updated, shifted coeffs
    ustar[...] = rd[...] * vmag[...]
    tstar[...] = rh[...] * delt[...]
    qstar[...] = re[...] * delq[...]

    # Compute the fluxes

    tau = rbot[...] * ustar[...] * ustar[...]

    # momentum flux
    taux = tau[...] * ubot[...] / vmag[...] * mask[...]
    tauy = tau[...] * vbot[...] / vmag[...] * mask[...]

    # heat flux
    sen = cp[...] * tau[...] * tstar[...] / ustar[...] * mask[...]
    lat = ct.LTHEAT * tau[...] * qstar[...] / ustar[...] * mask[...]
    lwup = -ct.STEBOL * ts[...]**4 * mask[...]

    # water flux
    evap = lat[...] / ct.LTHEAT * mask[...]

    # compute diagnostic: 2m reference height temperature

    # compute function of exchange coefficients. Assume that
    # cn = rdn*rdn, cm=rd*rd and ch=rh*rd, and therefore
    # 1/sqrt(cn(n))=1/rdn and sqrt(cm(n))/ch(n)=1/rh
    bn = ct.KARMAN / rdn
    bh = ct.KARMAN / rh[...]

    # interpolation factor for stable and unstable cases
    ln0 = npx.log(1.0 + (ct.ZTREF / zbot[...]) * (npx.exp(bn) - 1.0))
    ln3 = npx.log(1.0 + (ct.ZTREF / zbot[...]) * (npx.exp(bn - bh[...]) - 1.0))
    fac = (ln0[...] - ct.ZTREF/zbot[...] * (bn - bh[...])) / bh[...] * stable[...]\
        + (ln0[...] - ln3[...]) / bh[...] * (1.0 - stable[...])
    fac = npx.minimum(npx.maximum(fac, 0.0), 1.0)

    # actual interpolation
    tref = (ts[...] + (tbot[...] - ts[...]) * fac[...]) * mask[...]
    qref = (qbot[...] - delq[...] * fac[...]) * mask[...]

    return (sen, lat, lwup, evap, taux, tauy, tref, qref)


def flux_atmOcn(mask, rbot, zbot, ubot, vbot, qbot, tbot, thbot, us, vs, ts):
    """atm/ocn fluxes calculation

    Arguments:
        mask (:obj:`ndarray`): ocn domain mask       0 <=> out of domain
        rbot (:obj:`ndarray`): atm density           (kg/m^3)
        zbot (:obj:`ndarray`): atm level height      (m)
        ubot (:obj:`ndarray`): atm u wind            (m/s)
        vbot (:obj:`ndarray`): atm v wind            (m/s)
        qbot (:obj:`ndarray`): atm specific humidity (kg/kg)
        tbot (:obj:`ndarray`): atm T                 (K)
        thbot(:obj:`ndarray`): atm potential T       (K)
        us   (:obj:`ndarray`): ocn u-velocity        (m/s)
        vs   (:obj:`ndarray`): ocn v-velocity        (m/s)
        ts   (:obj:`ndarray`): ocn temperature       (K)

    Returns:
        sen  (:obj:`ndarray`): heat flux: sensible    (W/m^2)
        lat  (:obj:`ndarray`): heat flux: latent      (W/m^2)
        lwup (:obj:`ndarray`): heat flux: lw upward   (W/m^2)
        evap (:obj:`ndarray`): water flux: evap  ((kg/s)/m^2)
        taux (:obj:`ndarray`): surface stress, zonal      (N)
        tauy (:obj:`ndarray`): surface stress, maridional (N)

        tref (:obj:`ndarray`): diag:  2m ref height T     (K)
        qref (:obj:`ndarray`): diag:  2m ref humidity (kg/kg)
        duu10n(:obj:`ndarray`): diag: 10m wind speed squared (m/s)^2

        ustar_sv(:obj:`ndarray`): diag: ustar
        re_sv   (:obj:`ndarray`): diag: sqrt of exchange coefficient (water)
        ssq_sv  (:obj:`ndarray`): diag: sea surface humidity  (kg/kg)

    Reference:
        - Large, W. G., & Pond, S. (1981). Open Ocean Momentum Flux Measurements in Moderate to Strong Winds,
        Journal of Physical Oceanography, 11(3), pp. 324-336
        - Large, W. G., & Pond, S. (1982). Sensible and Latent Heat Flux Measurements over the Ocean,
        Journal of Physical Oceanography, 12(5), 464-482.
        - https://svn-ccsm-release.cgd.ucar.edu/model_versions/cesm1_0_5/models/csm_share/shr/shr_flux_mod.F90
    """

    al2 = npx.log(ct.ZREF / ct.ZTREF)

    vmag = npx.maximum(ct.UMIN_O, npx.sqrt((ubot[...] - us[...])**2
                                        + (vbot[...] - vs[...])**2))

    # sea surface humidity (kg/kg)
    ssq = 0.98 * qsat(ts[...]) / rbot[...]

    # potential temperature diff. (K)
    delt = thbot[...] - ts[...]

    # specific humidity diff. (kg/kg)
    delq = qbot[...] - ssq[...]

    alz = npx.log(zbot[...] / ct.ZREF)
    cp = ct.CPDAIR * (1.0 + ct.CPVIR * ssq[...])

    # first estimate of Z/L and ustar, tstar and qstar

    # neutral coefficients, z/L = 0.0
    stable = 0.5 + 0.5 * npx.sign(delt[...])
    rdn = npx.sqrt(cdn(vmag[...]))
    rhn = (1.0 - stable) * 0.0327 + stable * 0.018
    ren = 0.0346

    ustar = rdn * vmag[...]
    tstar = rhn * delt[...]
    qstar = ren * delq[...]

    # compute stability & evaluate all stability functions
    hol = ct.KARMAN * ct.G * zbot[...] *\
        (tstar[...] / thbot[...] + qstar[...]
         / (1.0 / ct.ZVIR + qbot[...])) / ustar[...]**2
    hol[...] = npx.minimum(npx.abs(hol[...]), 10.0) * npx.sign(hol[...])
    stable = 0.5 + 0.5 * npx.sign(hol[...])
    xsq = npx.maximum(npx.sqrt(npx.abs(1.0 - 16.0 * hol[...])), 1.0)
    xqq = npx.sqrt(xsq[...])
    psimh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq[...])
    psixh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])

    # shift wind speed using old coefficient
    rd = rdn[...] / (1.0 + rdn[...] / ct.KARMAN * (alz[...] - psimh[...]))
    u10n = vmag[...] * rd[...] / rdn[...]

    # update transfer coeffs at 10m and neutral stability
    rdn = npx.sqrt(cdn(u10n[...]))
    ren = 0.0346
    rhn = (1.0 - stable[...]) * 0.0327 + stable[...] * 0.018

    # shift all coeffs to measurement height and stability
    rd = rdn[...] / (1.0 + rdn[...] / ct.KARMAN * (alz[...] - psimh[...]))
    rh = rhn[...] / (1.0 + rhn[...] / ct.KARMAN * (alz[...] - psixh[...]))
    re = ren / (1.0 + ren / ct.KARMAN * (alz[...] - psixh[...]))

    # update ustar, tstar, qstar using updated, shifted coeffs
    ustar = rd[...] * vmag[...]
    tstar = rh[...] * delt[...]
    qstar = re[...] * delq[...]

    # iterate to converge on Z/L, ustar, tstar and qstar

    # compute stability & evaluate all stability functions
    hol = ct.KARMAN * ct.G * zbot[...] *\
        (tstar[...] / thbot[...] + qstar[...]
         / (1.0 / ct.ZVIR + qbot[...])) / ustar[...]**2
    hol[...] = npx.minimum(npx.abs(hol[...]), 10.0) * npx.sign(hol[...])
    stable[...] = 0.5 + 0.5 * npx.sign(hol[...])
    xsq[...] = npx.maximum(npx.sqrt(npx.abs(1.0 - 16.0 * hol[...])), 1.0)
    xqq[...] = npx.sqrt(xsq[...])
    psimh[...] = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq[...])
    psixh[...] = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])

    # shift wind speed using old coefficient
    rd[...] = rdn[...] / (1.0 + rdn[...] / ct.KARMAN * (alz[...] - psimh[...]))
    u10n = vmag[...] * rd[...] / rdn[...]

    # update transfer coeffs at 10m and neutral stability
    rdn[...] = npx.sqrt(cdn(u10n[...]))
    ren = 0.0346
    rhn[...] = (1.0 - stable[...]) * 0.0327 + stable[...] * 0.018

    # shift all coeffs to measurement height and stability
    rd[...] = rdn[...] / (1.0 + rdn[...] / ct.KARMAN * (alz[...] - psimh[...]))
    rh[...] = rhn[...] / (1.0 + rhn[...] / ct.KARMAN * (alz[...] - psixh[...]))
    re[...] = ren / (1.0 + ren / ct.KARMAN * (alz[...] - psixh[...]))

    # update ustar, tstar, qstar using updated, shifted coeffs
    ustar[...] = rd[...] * vmag[...]
    tstar[...] = rh[...] * delt[...]
    qstar[...] = re[...] * delq[...]

    # compute the fluxes

    tau = rbot[...] * ustar[...] * ustar[...]

    # momentum flux
    taux = tau[...] * (ubot[...] - us[...]) / vmag[...] * mask[...]
    tauy = tau[...] * (vbot[...] - vs[...]) / vmag[...] * mask[...]

    # heat flux
    sen = cp[...] * tau[...] * tstar[...] / ustar[...] * mask[...]
    lat = ct.LATVAP * tau[...] * qstar[...] / ustar[...] * mask[...]
    lwup = -ct.STEBOL * ts[...]**4 * mask[...]

    # water flux
    evap = lat[...] / ct.LATVAP * mask[...]

    # compute diagnositcs: 2m ref T & Q, 10m wind speed squared

    hol[...] = hol[...] * ct.ZTREF / zbot[...]
    xsq = npx.maximum(1.0, npx.sqrt(npx.abs(1.0 - 16.0 * hol[...])))
    xqq = npx.sqrt(xsq)
    psix2 = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])
    fac = (rh[...] / ct.KARMAN) * (alz[...] + al2 - psixh[...] + psix2[...])
    tref = thbot[...] - delt[...] * fac[...]

    # pot. temp to temp correction
    tref[...] = (tref[...] - 0.01 * ct.ZTREF) * mask[...]
    fac[...] = (re[...] / ct.KARMAN) * (alz[...] + al2 - psixh[...] + psix2[...]) * mask[...]
    qref = (qbot[...] - delq[...] * fac[...]) * mask[...]

    # 10m wind speed squared
    duu10n = u10n[...] * u10n[...] * mask[...]

    return (sen, lat, lwup, evap, taux, tauy, tref, qref, duu10n)


def main(state):
    vs = state.variables

    # read netcdf files
    def read_forcing(var, file):
        with netCDF4.Dataset(file) as infile:
            return npx.squeeze(infile[var][:].T)

    year_in_seconds = 360 * 86400.0
    (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(vs.time, year_in_seconds, year_in_seconds / 12.0, 12)

    # interpolate the monthly mean data to the value at the current time step
    def current_value(field):
        return f1 * field[:, :, n1] + f2 * field[:, :, n2]

    def current_value4d(field):
        return f1 * field[:, :, :, n1] + f2 * field[:, :, :, n2]


    # def plot(ds, fname, cmap=None, vmin=None, vmax=None):
    #     plt.figure(figsize=(10, 5))
    #     cs = plt.pcolor(longitude, latitude,
    #                     ds.T, cmap=cmap,
    #                     vmin=vmin, vmax=vmax,
    #                     shading='auto')
    #     plt.colorbar(cs)
    #     plt.savefig(f'{output_path}/{fname}.png')

    # Fields in the input file are defined on IFS model levels:
    #   levels 0 (L1)   - surfaces (lnsp & z)
    #   levels 1 (L136) - bottom - 1
    #   levels 2 (L137) - bottom
    #
    # Sigma coefficients hyam(i) & hybm(i) are given on
    #   L1 (TOA) - L137(8) (near-surface) model levels
    PATH = '/Users/jgaertne/Documents/forcing data/'
    DATA_ML = './era5_198x_ml_4x4deg_monthly_mean.nc'
    DATA_SFC = './era5_198x_sfc_4x4deg_monthly_mean.nc'
    input_era5_ml = PATH + DATA_ML
    longitude = read_forcing('longitude', input_era5_ml)
    latitude = read_forcing('latitude', input_era5_ml)
    hyai = read_forcing('hyai', input_era5_ml)[-3:]
    hybi = read_forcing('hybi', input_era5_ml)[-3:]
    hyam = read_forcing('hyam', input_era5_ml)[-2:]   # L136-L137
    hybm = read_forcing('hybm', input_era5_ml)[-2:]   # L136-L137
    lnsp = current_value(read_forcing('lnsp', input_era5_ml)[..., 0, :])
    ubot = current_value(read_forcing('u', input_era5_ml)[..., 1, :])  # L136
    vbot = current_value(read_forcing('v', input_era5_ml)[..., 1, :])  # L136
    q = current_value4d(read_forcing('q', input_era5_ml)[..., 1:, :])     # L136-L137
    t = current_value4d(read_forcing('t', input_era5_ml)[..., 1:, :])     # L136-L137

    qbot = q[..., 0]   # L136
    tbot = t[..., 0]   # L136
    input_era5_sfc = PATH + DATA_SFC
    lsm = current_value(read_forcing('lsm', input_era5_sfc))
    siconc = current_value(read_forcing('siconc', input_era5_sfc))
    sst = current_value(read_forcing('sst', input_era5_sfc))
    tcc = current_value(read_forcing('tcc', input_era5_sfc))
    swr_net = current_value(read_forcing('msnswrf', input_era5_sfc))
    lwr_net = current_value(read_forcing('msnlwrf', input_era5_sfc))

    # veros and forcing grid
    t_grid = (vs.xt[2:-2], vs.yt[2:-2])
    xt_forc = npx.array(netCDF4.Dataset(PATH + DATA_ML)['longitude'])
    yt_forc = npx.array(netCDF4.Dataset(PATH + DATA_ML)['latitude'][::-1])
    forc_grid = (xt_forc, yt_forc)

    # interpolate veros variables to forcing grid
    def interpolate(var):
        return veros.tools.interpolate(t_grid, var, forc_grid)

    # copy ocean temperature and velocity
    ts = interpolate(vs.temp[2:-2,2:-2,-1,vs.tau])
    us = interpolate(vs.u[2:-2,2:-2,-1,vs.tau])
    vs = interpolate(vs.v[2:-2,2:-2,-1,vs.tau])

    sp = npx.exp(lnsp)
    ph = get_press_levs(sp, hyai, hybi)
    pf = get_press_levs(sp, hyam, hybm)
    zbot = compute_z_level(t, q, ph)   # L136

    # air density
    rbot = ct.MWDAIR / ct.RGAS * pf[:, :, 0] / tbot[...]   # L136

    # potential temperature
    thbot = (tbot[...] * (ct.P0 / pf[:, :, 0])**ct.CAPPA)    # L136

    mask_nan = npx.isnan(ts)
    mask_ice = npx.zeros(siconc.shape)
    mask_ocn = npx.zeros(lsm.shape)
    mask_ice = npx.zeros(lsm.shape)

    ts = update(ts, at[mask_nan], 0)
    us = update(us, at[mask_nan], 0)
    vs = update(vs, at[mask_nan], 0)

    # ice mask (1 if there is ice)
    mask_ice[siconc > 0.] = 1
    # ocean mask (1 in the ocean, 0 on land)
    mask_ocn[lsm == 0.] = 1
    # ocean mask without ice (1 in the ocean, 0 on land and if there is ice)
    mask_ocn_ice = mask_ocn.copy()
    mask_ocn_ice[siconc > 0.] = 0

    atmOcn_fluxes =\
        dict(zip(('sen', 'lat', 'lwup', 'evap', 'taux', 'tauy', 'tref', 'qref', 'duu10n'),
             flux_atmOcn(mask_ocn_ice, rbot, zbot, ubot, vbot, qbot, tbot, thbot, us, vs, ts)))

    atmIce_fluxes =\
        dict(zip(('sen', 'lat', 'lwup', 'evap', 'taux', 'tauy', 'tref', 'qref'),
             flux_atmIce(mask_ice, rbot, zbot, ubot, vbot, qbot, tbot, thbot, ts)))

    # Net LW radiation flux from sea surface
    lwnet_ocn = net_lw_ocn(mask_ocn_ice, latitude, qbot, sst, tbot, tcc)

    # Downward LW radiation flux over sea-ice
    lwdw_ice = dw_lw_ice(mask_ice, tbot, tcc)

    # Net surface radiation flux (without short-wave)
    qnet = -(swr_net + lwnet_ocn
         + lwdw_ice + atmIce_fluxes['lwup']
         + atmIce_fluxes['sen'] + atmOcn_fluxes['sen']
         + atmIce_fluxes['lat'] + atmOcn_fluxes['lat'])

    dqir_dt, dqh_dt, dqe_dt = dqnetdt(mask_ocn, sp, rbot, sst, ubot, vbot, us, vs)

    # ----------------------------------------------------------------

    qnec = - ( dqir_dt + dqh_dt + dqe_dt )

    return qnet, qnec

    # output_path = './output'
    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)

    # plot((swr_net + lwr_net) * mask_ocn, 'swr_lwr_net')
    # plot(lwnet_ocn, 'lwnet_ocn')
    # plot(lwdw_ice, 'lwdw_ice')
    # plot(qnet, 'qnet', cmap='RdBu_r', vmin=-400, vmax=400)
    # plot(-(dqir_dt + dqh_dt + dqe_dt), 'dqnet_dt', vmin=0, vmax=70)

    # for fld in atmIce_fluxes:
    #     plot(atmIce_fluxes[fld] + atmOcn_fluxes[fld], fld)


if __name__ == "__main__":
    main(0)
