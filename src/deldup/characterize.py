import gzip
import csv
import os

import pandas as pd

class Experiment():
    def __init__(self, source):
        self.outpath = ''
        self.data = self.extract_data(source)

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
        pass
