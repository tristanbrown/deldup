import gzip
import csv
import os

import numpy as np
import pandas as pd

from .stats import model_cn, fit_gaussian, cluster, cluster_bisect, get_bounds
from .breakpoint import extract_region

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

        # Hardcoded
        self.ethnicities = ['A', 'B', 'C']
        self.breakpoints = pd.DataFrame(
            [[32, 38], [27, 34], [20, 40], [10, 40]],
            index=range(1,5), columns=['5-break', '3-break'])

        # Outputs
        self.probe_devs = None
        self.data_norm = None
        self.data_cn = None
        self.deldup_counts = None
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
        bounds = get_bounds(self.models)
        self.data_cn = self.categorize_cn(bounds)
        self.deldup_counts = self.count_cn(self.data_cn)
        self.sample_groups = {label: SampleGroup(label, self.deldup_counts)
            for label in self.ethnicities}
        self.extract_all_regions()
        self.final_tbl = self.tabulate_groups()
        self.save_outputs()

    def tabulate_groups(self):
        rows = [sample.get_row() for sample in self.sample_groups.values()]
        return pd.concat(rows, axis=1).T

    def extract_all_regions(self):
        for sample in self.sample_groups.values():
            deldups, residual = extract_region(sample.raw_counts, self.breakpoints)
            deldups = pd.DataFrame(deldups, columns=['deldup_idx', 'Del', 'Dup'])
            sample.deldups = deldups.set_index('deldup_idx').sort_index()
            sample.residual = residual[(residual['Del'] != 0) | (residual['Dup'] != 0)]

    def count_cn(self, data):
        return data.groupby('ethnicity').agg(
            {i:'value_counts' for i in self.probe_cols})

    def categorize_cn(self, bounds):
        data_cn = self.data[['ethnicity']].merge(
            self.data_norm, how='left', left_index=True, right_index=True)
        for col, bound in bounds.items():
            conditions = (
                data_cn[col] < bound[0],
                (data_cn[col] > bound[0])&(data_cn[col] < bound[1]),
                data_cn[col] > bound[1])
            data_cn[col] = np.select(conditions, (1, 2, 3), default=0)
        return data_cn

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
        self.final_tbl.to_csv(os.path.join(self.outpath, 'deldup_percentages.csv'))

class SampleGroup():
    def __init__(self, label, source):
        self.label = label
        self.raw_counts = source.loc[label,:].T.rename(
            {0: 'Bad data', 1: 'Del', 2: 'Normal', 3: 'Dup'}, axis=1)
        self.raw_counts = self.raw_counts.fillna(0)
        self.total = self.raw_counts.sum(axis=1).mode()[0]
        self.deldups = None
        self.residual = None

    def get_row(self):
        perc = self.deldups / self.total * 100
        print(perc)
        row_data = perc.stack()
        row_data.name = self.label
        print(row_data)
        return row_data
