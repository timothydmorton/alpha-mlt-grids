import pandas as pd
import numpy as np

from isochrones.priors import EEP_prior


class AlphaMLT_EEP_prior(EEP_prior):
    def _pdf(self, eep, **kwargs):
        pars = [kwargs["mass"], eep, kwargs["feh"], kwargs["alpha_mlt"]]
        orig_val, dx_deep = self.ic.interp_value(
            pars, [self.orig_par, self.deriv_prop]
        ).squeeze()
        return self.orig_prior(orig_val) * dx_deep

    def sample(self, n, **kwargs):
        raise NotImplemented

        # adapt the below if needed
        eeps = pd.Series(np.arange(self.bounds[0], self.bounds[1])).sample(
            n, replace=True
        )

        if self.orig_par == "age":
            mass = kwargs["mass"]
            if isinstance(mass, pd.Series):
                mass = mass.values
            feh = kwargs["feh"]
            if isinstance(feh, pd.Series):
                feh = feh.values
            values = self.ic.interp_value([mass, eeps, feh], ["dt_deep", "age"])
            deriv_val, orig_val = values[:, 0], values[:, 1]
            orig_pr = np.array([self.orig_prior(v) for v in orig_val])
            # weights = orig_pr * np.log10(orig_val)/np.log10(np.e) * deriv_val  # why like this?
            weights = orig_pr * deriv_val
        elif self.orig_par == "mass":
            age = kwargs["age"]
            if isinstance(age, pd.Series):
                age = age.values
            feh = kwargs["feh"]
            if isinstance(feh, pd.Series):
                feh = feh.values
            values = self.ic.interp_value([eeps, age, feh], ["dm_deep", "mass"])
            deriv_val, orig_val = values[:, 0], values[:, 1]
            orig_pr = np.array([self.orig_prior(v) for v in orig_val])
            weights = orig_pr * deriv_val

        try:
            return eeps.sample(n, weights=weights, replace=True).values
        except ValueError:
            # If there are no valid samples, just run it again until you get valid results
            return self.sample(n, **kwargs)
