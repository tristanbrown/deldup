import gzip
import csv
import os

import pandas as pd

from .stats import model_cn, fit_gaussian, cluster, cluster_bisect

class Experiment():
    def __init__(self, source):
        # I/O
        self.outpath = ''
        self.data = self.extract_data(source)

        # Groups of column names
        self.probe_cols = [col for col in self.data.columns if 'probe' in col]
        self.ctrl_cols, self.regn_cols = [], []
        for col in self.probe_cols:
            self.ctrl_cols.append(col) if col.startswith('non') else self.regn_cols.append(col)

        # Outputs
        self.probe_devs = None
        self.data_norm = None
        self.metrics = None
        self.models = None

    def extract_data(self, source):
        if isinstance(source, pd.DataFrame):
            return source
        elif isinstance(source, str):
            data = pd.read_csv(source, index_col=[0])
            self.outpath = os.path.dirname(source)
            return data
        else:
            raise TypeError(
                "Unknown input type. Please provide a DataFrame or path to a csv file.")

    def analyze(self):
        self.data = self.depth_factor(self.data)
        self.data_norm, self.metrics = self.normalize_read_depth(self.data)
        self.models = {col: model_cn(self.data_norm[col]) for col in self.probe_cols}
        self.save_outputs()

    def fit_models(self, names=None):
        if not names:
            names = self.probe_cols
        for col in names:
            print(col)
            fit_gaussian(self.models[col], visual=True)

    def cluster_all(self, names=None):
        if not names:
            names = self.probe_cols
        for col in names:
            print(col)
            clusters = cluster(self.data_norm[col], k=3)
            fit_gaussian(clusters, visual=True)

    def cluster_bisect_all(self, names=None):
        if not names:
            names = self.probe_cols
        for col in names:
            print(col)
            try:
                clusters = cluster_bisect(self.data_norm[col])
                fit_gaussian(clusters, visual=True)
            except ValueError:
                print("FAIL")

    def depth_factor(self, data):
        """Calculate a per-sample normalization factor for read depth."""
        depth_dict = {}
        for col in self.ctrl_cols:
            depth_dict[f'{col}_factor'] = self.data[col] / self.data[col].mean()
        depth_df = pd.DataFrame(depth_dict)
        data['depth_factor'] = depth_df.mean(axis=1)
        data['depth_factor_std'] = depth_df.std(axis=1)
        self.probe_devs = depth_df.std(axis=0).sort_values(ascending=False)
        return data
    
    def normalize_read_depth(self, data):
        """Normalize the read depth."""
        normalized = data[self.probe_cols].div(data['depth_factor'], axis=0)
        desc = normalized.describe()
        desc = desc.append(pd.Series(desc.loc['std',:]/desc.loc['mean',:],name='coef_var'))
        return normalized, desc

    def save_outputs(self):
        """Save the outputs to files."""
        print(self.data.head())
        self.data.to_csv(os.path.join(self.outpath, 'out_data.csv'))
        self.probe_devs.to_csv(os.path.join(self.outpath, 'out_stats.csv'), header=False)
