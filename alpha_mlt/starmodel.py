import numpy as np
import pandas as pd

from isochrones.starmodel import BasicStarModel
from isochrones.likelihood import star_lnlike_4d, gauss_lnprob
from isochrones.priors import FlatPrior

from .prior import AlphaMLT_EEP_prior

class AlphaMLTStarModel(BasicStarModel):

    physical_quantities = [
        "mass",
        "radius",
        "age",
        "Teff",
        "logg",
        "feh",
        "alpha_mlt",
        "distance",
        "AV",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.N == 1:
            self.mass_index = 0
            self.feh_index = 2
            self.alpha_mlt_index = 3
            self.distance_index = 4
            self.AV_index = 5
        elif self.N == 2:
            raise ValueError("Need to build alpha_mlt isochrones before using this.")
        elif self.N == 3:
            raise ValueError("Need to build alpha_mlt isochrones before using this.")

        self._priors["eep"] = AlphaMLT_EEP_prior(
            self.ic, self._priors[self.ic.eep_replaces], bounds=self.eep_bounds
        )
        self._priors["alpha_mlt"] = FlatPrior(self.ic.alpha_mlt_bounds)

        self._bounds["alpha_mlt"] = self.ic.alpha_mlt_bounds

    def lnlike(self, pars):
        if self.N == 1:
            pars = np.array([pars[0], pars[1], pars[2], pars[3], pars[4], pars[5]], dtype=float)
            primary_pars = pars
        elif self.N == 2:
            primary_pars = np.array([pars[0], pars[2], pars[3], pars[4], pars[5], pars[6]])
            pars = np.array([pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6]], dtype=float,)
        elif self.N == 3:
            primary_pars = np.array([pars[0], pars[3], pars[4], pars[5], pars[6], pars[7]])
            pars = np.array(
                [pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7],], dtype=float,
            )

        spec_vals, spec_uncs = zip(*[prop for prop in self.spec_props])
        if self.bands:
            mag_vals, mag_uncs = zip(*[self.kwargs[b] for b in self.bands])
            i_mags = [self.ic.bc_grid.interp.column_index[b] for b in self.bands]
        else:
            mag_vals, mag_uncs = np.array([], dtype=float), np.array([], dtype=float)
            i_mags = np.array([], dtype=int)
        lnlike = star_lnlike_4d(
            pars,
            self.ic.param_index_order,
            spec_vals,
            spec_uncs,
            mag_vals,
            mag_uncs,
            i_mags,
            self.ic.model_grid.interp.grid,
            self.ic.model_grid.interp.column_index["Teff"],
            self.ic.model_grid.interp.column_index["logg"],
            self.ic.model_grid.interp.column_index["feh"],
            self.ic.model_grid.interp.column_index["Mbol"],
            *self.ic.model_grid.interp.index_columns,
            self.ic.bc_grid.interp.grid,
            *self.ic.bc_grid.interp.index_columns,
        )

        if "parallax" in self.kwargs:
            plax, plax_unc = self.kwargs["parallax"]
            lnlike += gauss_lnprob(plax, plax_unc, 1000.0 / pars[self.distance_index])

        # Asteroseismology
        if "nu_max" in self.kwargs:
            model_nu_max, model_delta_nu = self.ic.interp_value(primary_pars, ["nu_max", "delta_nu"])

            nu_max, nu_max_unc = self.kwargs["nu_max"]
            lnlike += gauss_lnprob(nu_max, nu_max_unc, model_nu_max)

            if "delta_nu" in self.kwargs:
                delta_nu, delta_nu_unc = self.kwargs["delta_nu"]
                lnlike += gauss_lnprob(delta_nu, delta_nu, model_delta_nu)

        return lnlike

    def lnprior(self, pars):
        lnp = 0
        if self.N == 2:
            if pars[1] > pars[0]:
                return -np.inf
        elif self.N == 3:
            if not (pars[0] > pars[1]) and (pars[1] > pars[2]):
                return -np.inf
        for val, par in zip(pars, self.param_names):
            if par in ["eep", "eep_0", "eep_1", "eep_2"]:
                lnp += self._priors["eep"].lnpdf(
                    val,
                    mass=pars[self.mass_index],
                    feh=pars[self.feh_index],
                    alpha_mlt=pars[self.alpha_mlt_index],
                )
            else:
                lnp += self._priors[par].lnpdf(val)

        return lnp

    def _make_samples(self):
        filename = "{}post_equal_weights.dat".format(self.mnest_basename)
        try:
            df = pd.read_csv(filename, names=self.param_names + ("lnprob",), delim_whitespace=True)
        except OSError:
            logger.error("Error loading chains from {}".format(filename))
            raise

        self._samples = df

        if self.N == 1:
            self._derived_samples = self.ic(*[df[c].values for c in self.param_names])

        self._derived_samples["parallax"] = 1000.0 / df["distance"]
        self._derived_samples["distance"] = df["distance"]
        self._derived_samples["AV"] = df["AV"]
