import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px
import warnings
warnings.simplefilter("ignore", FutureWarning)


def CV(data, log2=False, threshold=30, qc="_NIST_", filter_by='nonparametric', apply_filter=False):
    qc_data = data[data.index.str.contains(qc)].apply(pd.to_numeric)
    if log2:
        ss = qc_data.var(axis=0, ddof=1, skipna=True)
        cv_param = 100 * np.sqrt(np.exp((np.log(2)**2) * ss) - 1)
        cv_nonparam = cv_param
    else:
        cv_param = 100 * (qc_data.std(axis=0, ddof=1) / qc_data.mean(axis=0))
        mad = 1.4826 * qc_data.apply(lambda x: np.median(np.abs(x - np.median(x))))
        cv_nonparam = (mad / qc_data.median()) * 100
    cv_table = pd.DataFrame({
        'CV_parametric': cv_param,
        'CV_nonparametric': cv_nonparam
    })
    if not apply_filter:
        return cv_table
    else:
        col = 'CV_parametric' if filter_by == 'parametric' else 'CV_nonparametric'
        selected = cv_table[cv_table[col] <= threshold].index
        return data.loc[:, selected]


def D_ratio(data, qc="_NIST_", biological='_S_', threshold=50, filter_by='nonparametric', apply_filter=False):
    qc_data = data[data.index.str.contains(qc)].apply(pd.to_numeric)
    sample_data = data[data.index.str.contains(biological)].apply(pd.to_numeric)
    d_param = (qc_data.std(ddof=1) / sample_data.std(ddof=1)) * 100
    mad_qc = qc_data.apply(lambda x: np.median(np.abs(x - np.median(x))))
    mad_bio = sample_data.apply(lambda x: np.median(np.abs(x - np.median(x))))
    d_nonparam = (mad_qc / mad_bio) * 100
    d_table = pd.DataFrame({
        'D_ratio_parametric': d_param,
        'D_ratio_nonparametric': d_nonparam
    })
    if not apply_filter:
        return d_table
    else:
        col = 'D_ratio_parametric' if filter_by == 'parametric' else 'D_ratio_nonparametric'
        selected = d_table[d_table[col] <= threshold].index
        return data.loc[:, selected]


def detection_rate(data,qc='_NIST_'):
    qc_data = data[data.index.str.contains(qc)].apply(pd.to_numeric)
    detection = 1 - (qc_data.isna().sum(axis=0) / qc_data.shape[0])
    detection_rate = detection * 100
    return detection_rate

