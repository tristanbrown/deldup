import gzip
import csv
import os

import pandas as pd

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
        print(self.data.head())
        self.data.to_csv(os.path.join(self.outpath, 'out_data.csv'))
        self.probe_devs.to_csv(os.path.join(self.outpath, 'out_stats.csv'))

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
