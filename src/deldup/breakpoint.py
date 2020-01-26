"""Functions for handling the breakpoints table."""
import pandas as pd

def probe_names(alist):
    return [f"CNSL_probe_{n}" for n in alist]

def region_probes(tbl, idx):
    return probe_names(range(tbl.loc[idx,'5-break'], tbl.loc[idx, '3-break']+1))

def choose_region(tbl):
    tbl = tbl.copy()
    tbl['length'] = tbl['3-break'] - tbl['5-break']
    idx = tbl[tbl['length'] == tbl['length'].max()].index[0]
    return int(idx)

def get_overhang(tbl, idx):
    reg = tbl.loc[idx,:]
    tbl = tbl.copy().drop(idx, axis=0)
    if tbl.empty:
        return probe_names(range(reg[0], reg[1]+1))
    left_probes = list(range(reg[0], tbl['5-break'].min()))
    right_probes = list(range(tbl['3-break'].max()+1, reg[1]+1))
    return probe_names(left_probes + right_probes)

def extract_region(eth_tbl, bp_tbl, region_deldups=None):
    if region_deldups is None:
        region_deldups = []
    if bp_tbl.empty:
        return region_deldups, eth_tbl.fillna(0)
    idx = choose_region(bp_tbl)
    overhang = get_overhang(bp_tbl, idx)
    dels, dups = [eth_tbl.loc[overhang, label].mode()[0] for label in ('Del', 'Dup')]
    region_deldups.append([idx, dels, dups])
    region = region_probes(bp_tbl, idx)
    reduced_tbl = eth_tbl.copy()
    reduced_tbl = subtract_index(reduced_tbl, 'Del', region, dels)
    reduced_tbl = subtract_index(reduced_tbl, 'Dup', region, dups)
    return extract_region(reduced_tbl, bp_tbl.drop(idx, axis=0), region_deldups)

def subtract_index(tbl, col, index, value):
    diff = pd.Series([value]*len(index), index=index)
    tbl['diff'] = diff
    tbl['diff'] = tbl['diff'].fillna(0)
    tbl[col] = tbl[col] - tbl['diff']
    return tbl.drop('diff', axis=1)
