"""
Tests for acceleration-only fits, ``System(acceleration_only=True)``: the
star's motion is position + proper motion + acceleration + parallax, with no
Keplerian orbit. Data can be Gaia DR4 epoch astrometry (through ``gaia``),
stellar ra/dec astrometry from any instrument (object 0 in the data table),
or both.
"""

import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time

from orbitize import DATADIR, gaia, hipparcos, plot, priors, read_input, sampler, system

REF_EPOCH_MJD = 57936.375  # J2017.5, the DR4 reference epoch [MJD]
ALPHA0_DEG, DELTA0_DEG = 86.82, -51.07  # reference RA/Dec for ra/dec data [deg]

# position, proper motion, acceleration and parallax used to make mock data
TRUTH = {
    "plx": 12.0,
    "pm_ra": 3.5,
    "pm_dec": -20.0,
    "alpha0": 0.4,
    "delta0": -0.2,
    "acc_ra": 0.15,
    "acc_dec": -0.08,
}

HIP_DATA = {
    "RArad": 86.82118072,
    "e_RArad": 0.1,
    "DErad": -51.06671341,
    "e_DErad": 0.11,
    "Plx": 51.44,
    "e_Plx": 0.12,
    "pmRA": 4.65,
    "e_pmRA": 0.11,
    "pmDE": 83.1,
    "e_pmDE": 0.15,
    "F2": -1.63,
    "Sn": 5,
    "var": 0.0,
}


def mock_dr4(truth=None, n_obs=40, err=0.05, jitter_prior=None, seed=0):
    """
    Mock Gaia DR4 along-scan data over 5 yr centred on J2017.5. If ``truth``
    is given, the AL positions are the noiseless acceleration model,
    computed here by hand:

        w = plx*f_AL + (alpha0 + pm_ra*dt + acc_ra*dt^2) sin(psi)
                     + (delta0 + pm_dec*dt + acc_dec*dt^2) cos(psi)
    """
    rng = np.random.default_rng(seed)
    epochs = REF_EPOCH_MJD + np.linspace(-2.5, 2.5, n_obs) * 365.25
    dt = (epochs - REF_EPOCH_MJD) / 365.25
    psi = rng.uniform(0, 2 * np.pi, n_obs)
    f_al = np.sin(2 * np.pi * dt)
    w = np.zeros(n_obs)
    if truth is not None:
        w = (
            truth["plx"] * f_al
            + (truth["alpha0"] + truth["pm_ra"] * dt + truth["acc_ra"] * dt**2) * np.sin(psi)
            + (truth["delta0"] + truth["pm_dec"] * dt + truth["acc_dec"] * dt**2) * np.cos(psi)
        )
    data = Table(
        {
            "obs_time_tcb": epochs,
            "centroid_pos_al": w,
            "centroid_pos_error_al": np.full(n_obs, err),
            "scan_pos_angle": psi,
            "parallax_factor_al": f_al,
            "field_of_view": np.zeros(n_obs, dtype=int),
        }
    )
    data.meta.update(time_format="mjd", time_scale="tcb")
    return gaia.DR4LogProb(
        0, data, ref_epoch_mjd=REF_EPOCH_MJD, jitter_prior=jitter_prior
    )


def radec_table(tmp_path, obj=0, years=(2015.0, 2017.0, 2018.5, 2020.0), err=0.5):
    """Write ra/dec astrometry of ``obj`` (0 = the star) at ``years`` to a csv
    file and read it back with ``read_input.read_file``."""
    epochs = Time(np.array(years), format="decimalyear").mjd
    n = len(epochs)
    t = Table(
        [epochs, np.full(n, obj), np.zeros(n), np.full(n, err), np.zeros(n), np.full(n, err)],
        names=["epoch", "object", "raoff", "raoff_err", "decoff", "decoff_err"],
    )
    filename = tmp_path / "astrometry.csv"
    t.write(filename, overwrite=True)
    return read_input.read_file(str(filename))


def acc_system(data_table=None, dr4=None, **kwargs):
    """Acceleration-only System. stellar_or_system_mass and plx are unused
    placeholders."""
    return system.System(
        num_secondary_bodies=0,
        data_table=data_table,
        stellar_or_system_mass=1.0,
        plx=10.0,
        gaia=dr4,
        acceleration_only=True,
        **kwargs,
    )


def params_from(sys, values):
    return np.array([values[label] for label in sys.labels], dtype=float)


def test_labels_and_priors():
    """The fitted parameters are plx, the DR4 astrometric parameters and the
    accelerations, in that order, with no orbit or mass parameters."""
    sys = acc_system(dr4=mock_dr4())

    assert sys.labels == ["plx", "pm_ra", "pm_dec", "alpha0", "delta0", "acc_ra", "acc_dec"]
    assert sys.param_idx == {label: i for i, label in enumerate(sys.labels)}
    assert len(sys.sys_priors) == len(sys.labels)
    assert all(isinstance(p, priors.UniformPrior) for p in sys.sys_priors)
    assert sys.acceleration_only
    assert not sys.acceleration_and_keplerian_fit


def test_labels_with_dr4_jitter():
    """A fitted DR4 jitter goes between the astrometric parameters and the
    accelerations."""
    sys = acc_system(dr4=mock_dr4(jitter_prior=priors.UniformPrior(0.0, 1.0)))

    assert sys.labels == [
        "plx", "pm_ra", "pm_dec", "alpha0", "delta0", "dr4_jitter", "acc_ra", "acc_dec",
    ]


def test_needs_data():
    """With no data table and no gaia object there is nothing to fit."""
    with pytest.raises(ValueError):
        acc_system()


def test_rejects_hipparcos():
    """Hipparcos IAD are not supported in an acceleration-only fit."""
    hip = hipparcos.HipparcosLogProb(
        "{}HIP027321.d".format(DATADIR), "027321", 0, hip_data=HIP_DATA
    )
    with pytest.raises(ValueError, match="hipparcos_IAD"):
        acc_system(dr4=mock_dr4(), hipparcos_IAD=hip)


def test_rejects_companion_astrometry(tmp_path):
    """Only stellar (object 0) ra/dec data can be used: there is no orbit to
    predict a companion's position."""
    with pytest.raises(ValueError, match="object 0"):
        acc_system(
            data_table=radec_table(tmp_path, obj=1),
            dr4=mock_dr4(),
            alpha0=ALPHA0_DEG,
            delta0=DELTA0_DEG,
        )


def test_radec_data_needs_reference_position(tmp_path):
    """Stellar ra/dec data need alpha0/delta0 to compute parallax factors."""
    with pytest.raises(ValueError, match="alpha0 and delta0"):
        acc_system(data_table=radec_table(tmp_path))


def test_rejects_both_fit_types():
    """acceleration_only and acceleration_and_keplerian_fit are exclusive."""
    with pytest.raises(ValueError):
        system.System(
            num_secondary_bodies=1,
            data_table=None,
            stellar_or_system_mass=1.0,
            plx=10.0,
            gaia=mock_dr4(),
            acceleration_only=True,
            acceleration_and_keplerian_fit=True,
        )


def test_star_offsets_are_acceleration_only():
    """compute_all_orbits gives no orbit: the star's offsets are -acc*dt^2
    (the sign DR4LogProb expects, see System._add_stellar_acceleration)."""
    dr4 = mock_dr4()
    sys = acc_system(dr4=dr4)
    params = params_from(sys, TRUTH)

    raoff, deoff, vz, brightness = sys.compute_all_orbits(params, epochs=dr4.epochs_mjd)

    assert raoff.shape == (dr4.n_obs, 1, 1)
    np.testing.assert_allclose(raoff[:, 0, 0], -TRUTH["acc_ra"] * dr4.dt_yr**2)
    np.testing.assert_allclose(deoff[:, 0, 0], -TRUTH["acc_dec"] * dr4.dt_yr**2)
    assert np.all(vz == 0) and np.all(brightness == 0)


def test_dr4_lnlike_matches_hand_calculation():
    """With noiseless mock data made from the model by hand, the residuals are
    zero at the true parameters, so lnlike is just the Gaussian normalisation.
    Moving the acceleration away from the truth lowers it."""
    dr4 = mock_dr4(truth=TRUTH)
    sys = acc_system(dr4=dr4)
    mcmc = sampler.MCMC(sys, num_temps=1, num_walkers=16, num_threads=1)
    params = params_from(sys, TRUTH)

    expected = -0.5 * np.sum(np.log(2 * np.pi * dr4.centroid_pos_error_al**2))
    np.testing.assert_allclose(mcmc._logl(params), expected, rtol=0, atol=1e-8)

    for label in ("acc_ra", "acc_dec"):
        off = params.copy()
        off[sys.param_idx[label]] += 0.01
        assert mcmc._logl(off) < expected


def test_least_squares_recovers_acceleration():
    """The AL model is linear in the parameters, so a least-squares fit to
    noiseless data built by hand recovers the injected values exactly. This
    checks that the acceleration has the same sign and dt^2 convention in the
    likelihood as in the hand-built data."""
    dr4 = mock_dr4(truth=TRUTH)
    sys = acc_system(dr4=dr4)

    def model(theta):
        return plot._model_along_scan_dr4(theta, sys, dr4)

    design = np.column_stack([model(e) for e in np.eye(len(sys.labels))])
    best, *_ = np.linalg.lstsq(design, dr4.centroid_pos_al, rcond=None)

    np.testing.assert_allclose(best, params_from(sys, TRUTH), rtol=0, atol=1e-8)


def test_radec_model_matches_hand_calculation(tmp_path):
    """compute_model for stellar ra/dec data: with zero parallax the model is
    the quadratic alpha0 + pm*dt + acc*dt^2 (RA divided by cos(delta0)),
    with dt measured from alphadec0_epoch."""
    years = np.array([2015.0, 2017.0, 2018.5, 2020.0])
    sys = acc_system(
        data_table=radec_table(tmp_path, years=years), alpha0=ALPHA0_DEG, delta0=DELTA0_DEG
    )
    values = dict(TRUTH, plx=0.0)
    model, jitter = sys.compute_model(params_from(sys, values))

    epochs_yr = Time(Time(years, format="decimalyear").mjd, format="mjd").decimalyear
    dt = epochs_yr - sys.alphadec0_epoch
    ra = values["alpha0"] + values["pm_ra"] * dt + values["acc_ra"] * dt**2
    dec = values["delta0"] + values["pm_dec"] * dt + values["acc_dec"] * dt**2

    np.testing.assert_allclose(model[:, 0], ra / np.cos(np.radians(DELTA0_DEG)))
    np.testing.assert_allclose(model[:, 1], dec)
    assert np.all(jitter == 0)


def test_radec_model_vectorised(tmp_path):
    """compute_model handles a 2D array of parameter vectors (one per
    column), as the samplers pass them in, and matches the 1D result."""
    sys = acc_system(data_table=radec_table(tmp_path), alpha0=ALPHA0_DEG, delta0=DELTA0_DEG)
    p1 = params_from(sys, TRUTH)
    p2 = params_from(sys, dict(TRUTH, acc_ra=-0.3, plx=5.0))

    model, _ = sys.compute_model(np.column_stack([p1, p2]))

    assert model.shape == (len(sys.data_table), 2, 2)
    np.testing.assert_allclose(model[..., 0], sys.compute_model(p1)[0])
    np.testing.assert_allclose(model[..., 1], sys.compute_model(p2)[0])


def test_radec_plus_dr4_lnlike_is_sum(tmp_path):
    """Fitting ra/dec astrometry and DR4 data together adds their
    log-likelihoods: the joint lnlike equals the sum of the two separate
    fits'."""
    table = radec_table(tmp_path)
    kwargs = dict(alpha0=ALPHA0_DEG, delta0=DELTA0_DEG)

    joint = acc_system(data_table=table, dr4=mock_dr4(truth=TRUTH), **kwargs)
    radec_only = acc_system(data_table=table, **kwargs)
    dr4_only = acc_system(dr4=mock_dr4(truth=TRUTH))
    assert joint.labels == radec_only.labels == dr4_only.labels

    params = params_from(joint, TRUTH)
    lnlike = lambda s: sampler.MCMC(s, num_temps=1, num_walkers=16, num_threads=1)._logl(params)

    np.testing.assert_allclose(lnlike(joint), lnlike(radec_only) + lnlike(dr4_only))


@pytest.mark.parametrize("use_radec, use_dr4", [(False, True), (True, False), (True, True)])
def test_mcmc_runs(tmp_path, use_radec, use_dr4):
    """A short MCMC run works for DR4 data, ra/dec data and both, and returns
    finite log-likelihoods."""
    kwargs = {}
    if use_radec:
        kwargs = dict(data_table=radec_table(tmp_path), alpha0=ALPHA0_DEG, delta0=DELTA0_DEG)
    sys = acc_system(dr4=mock_dr4(truth=TRUTH) if use_dr4 else None, **kwargs)

    n_walkers = 2 * len(sys.labels)
    mcmc = sampler.MCMC(sys, num_temps=1, num_walkers=n_walkers, num_threads=1)
    mcmc.curr_pos = params_from(sys, TRUTH) + 1e-3 * np.random.default_rng(0).normal(
        size=(1, n_walkers, len(sys.labels))
    )
    mcmc.run_sampler(n_walkers * 5, burn_steps=0)

    assert mcmc.results.post.shape == (n_walkers * 5, len(sys.labels))
    assert np.all(np.isfinite(mcmc.results.lnlike))


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    test_labels_and_priors()
    test_labels_with_dr4_jitter()
    test_needs_data()
    test_rejects_hipparcos()
    test_rejects_companion_astrometry(tmp)
    test_radec_data_needs_reference_position(tmp)
    test_rejects_both_fit_types()
    test_star_offsets_are_acceleration_only()
    test_dr4_lnlike_matches_hand_calculation()
    test_least_squares_recovers_acceleration()
    test_radec_model_matches_hand_calculation(tmp)
    test_radec_model_vectorised(tmp)
    test_radec_plus_dr4_lnlike_is_sum(tmp)
    test_mcmc_runs(tmp, True, True)
