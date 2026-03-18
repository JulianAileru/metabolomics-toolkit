import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.simplefilter("ignore", FutureWarning)
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis
    

def calc_kurtosis(data, qc="_NIST_"):
    qc_data = data[data.index.str.contains(qc)].apply(pd.to_numeric)
    return qc_data.apply(lambda x: scipy_kurtosis(x.dropna()))


def calc_skew(data, qc="_NIST_"):
    qc_data = data[data.index.str.contains(qc)].apply(pd.to_numeric)
    return qc_data.apply(lambda x: scipy_skew(x.dropna()))

def num_outliers(data, qc='_NIST_'):
    qc_data = data[data.index.str.contains(qc)].apply(pd.to_numeric)
    Q1 = qc_data.quantile(0.25)
    Q3 = qc_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (3 * IQR)
    upper_bound = Q3 + (3 * IQR)
    outliers = ((qc_data < lower_bound) | (qc_data > upper_bound)).sum()
    return outliers
def detection_rate(data,qc='_NIST_'):
    qc_data = data[data.index.str.contains(qc)].apply(pd.to_numeric)
    detection = 1 - (qc_data.isna().sum(axis=0) / qc_data.shape[0])
    detection_rate = detection * 100
    return detection_rate

def generate_stats(data, qc='_NIST_', biological='_S_',skew_thresh=2.0,kurt_thresh=2.0):
    cv_table = CV(data, qc=qc, apply_filter=False)
    d_table = D_ratio(data, qc=qc, biological=biological, apply_filter=False)
    kurt = calc_kurtosis(data, qc=qc)
    sk = calc_skew(data, qc=qc)
    outliers = num_outliers(data,qc=qc)
    stats = pd.concat([cv_table, d_table], axis=1)
    stats['kurtosis'] = kurt
    stats['skewness'] = sk
    stats['approx_normal'] = (sk.abs() < skew_thresh) & (kurt.abs() < kurt_thresh)
    stats['Number of Outliers in QC Samples'] = outliers
    return stats


