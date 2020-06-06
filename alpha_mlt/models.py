import glob
from pathlib import Path
import json
import re
import os

import numpy as np
import pandas as pd

from isochrones.models import StellarModelGrid, ModelGridInterpolator
from isochrones.mist.models import MISTEvolutionTrackGrid
from isochrones.mist.bc import MISTBolometricCorrectionGrid
from isochrones.mags import interp_mag_4d, interp_mags_4d
from isochrones.priors import FlatPrior


class AlphaMLTGrid(MISTEvolutionTrackGrid):

    name = "mist_mlt"

    index_cols = ("initial_feh", "initial_mass", "alpha_mlt", "EEP")
    ndim = 4

    filename_pattern = "\d+_\d+_\d+\.eep"

    feh_col = "initial_feh"

    default_columns = StellarModelGrid.default_columns + ("phase",)

    dataroot = Path("~/alpha_mlt/trimmed").expanduser()
    index = json.load(open(os.path.join(dataroot, "index.json")))

    fehs = np.array([f for f in index["feh"].values()])
    n_fehs = len(fehs)

    bounds = (
        ("eep", (0, 450)),
        ("age", (5, 10.3)),
        ("alpha_mlt", (0.31623, 3.16228)),
        ("feh", (-4.105, 0.395)),
        ("mass", (0.1, 0.775)),
    )

    @property
    def kwarg_tag(self):
        return "_test"

    @classmethod
    def parse_filename(cls, path):
        index = cls.index

        m = re.search("([0-9]+)_([0-9]+)_([0-9]+).eep", str(path))
        if m:
            mass = float(index["mass"][m.group(1)])
            feh = float(index["feh"][m.group(2)])
            alpha_mlt = float(index["alpha"][m.group(3)])
            return (mass, feh, alpha_mlt)
        else:
            raise ValueError(f"Cannot parse {path}!")

    @classmethod
    def get_mass(cls, filename):
        mass, _, _ = cls.parse_filename(filename)
        return mass

    @classmethod
    def get_feh(cls, filename):
        _, feh, _ = self.parse_filename(filename)
        return feh

    def get_directory_path(self):
        return self.dataroot

    def compute_surf_feh(self, df):
        return df["initial_feh"]  # Aaron Dotter says

    def df_all(self):
        return StellarModelGrid.df_all(self)

    @classmethod
    def to_df(cls, filename):
        df = super().to_df(filename)
        _, feh, alpha_mlt = cls.parse_filename(filename)
        df["alpha_mlt"] = alpha_mlt
        df["feh"] = feh
        df["initial_feh"] = feh
        return df


class AlphaMLTInterpolator(ModelGridInterpolator):
    grid_type = AlphaMLTGrid
    bc_type = MISTBolometricCorrectionGrid
    param_names = ("mass", "eep", "feh", "alpha_mlt", "distance", "AV")
    eep_bounds = (0, 450)
    alpha_mlt_bounds = (0.31623, 3.16228)
    eep_replaces = "age"

    # desired: mass, eep, feh, alpha_mlt, distance, AV
    _param_index_order = (
        2,
        0,
        3,
        1,
        4,
        5,
    )

    def __call__(self, p1, p2, p3, p4, distance=10.0, AV=0.0):
        p1, p2, p3, p4, dist, AV = [
            np.atleast_1d(a).astype(float).squeeze()
            for a in np.broadcast_arrays(p1, p2, p3, p4, distance, AV)
        ]
        pars = [p1, p2, p3, p4, dist, AV]
        # print(pars)
        prop_cols = self.model_grid.df.columns
        props = self.interp_value(pars, prop_cols)
        _, _, _, mags = self.interp_mag(pars, self.bands)
        cols = list(prop_cols) + ["{}_mag".format(b) for b in self.bands]
        values = np.concatenate([np.atleast_2d(props), np.atleast_2d(mags)], axis=1)
        df = pd.DataFrame(values, columns=cols)
        df["alpha_mlt"] = p4
        return df

    def interp_value(self, pars, props):
        """

        pars : age, feh, eep, [distance, AV]
        """
        try:
            pars = np.atleast_1d(pars[self.param_index_order])
        except TypeError:
            i0, i1, i2, i3, i4, i5 = self.param_index_order
            pars = [pars[i0], pars[i1], pars[i2], pars[i3]]
        #             print(pars)
        return self.model_grid.interp(pars, props)

    def interp_mag(self, pars, bands):
        """

        pars : age, feh, eep, distance, AV
        """
        if not bands:
            i_bands = np.array([], dtype=int)
        else:
            i_bands = [self.bc_grid.interp.columns.index(b) for b in bands]

        try:
            pars = np.atleast_1d(pars).astype(float).squeeze()
            if pars.ndim > 1:
                raise ValueError
            return interp_mag_4d(
                pars,
                self.param_index_order,
                self.model_grid.interp.grid,
                self.model_grid.interp.column_index["Teff"],
                self.model_grid.interp.column_index["logg"],
                self.model_grid.interp.column_index["feh"],
                self.model_grid.interp.column_index["Mbol"],
                *self.model_grid.interp.index_columns,
                self.bc_grid.interp.grid,
                i_bands,
                *self.bc_grid.interp.index_columns,
            )
        except (TypeError, ValueError):
            # Broadcast appropriately.
            b = np.broadcast(*pars)
            pars = np.array([np.resize(x, b.shape).astype(float) for x in pars])
            return interp_mags_4d(
                pars,
                self.param_index_order,
                self.model_grid.interp.grid,
                self.model_grid.interp.column_index["Teff"],
                self.model_grid.interp.column_index["logg"],
                self.model_grid.interp.column_index["feh"],
                self.model_grid.interp.column_index["Mbol"],
                *self.model_grid.interp.index_columns,
                self.bc_grid.interp.grid,
                i_bands,
                *self.bc_grid.interp.index_columns,
            )

