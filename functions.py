import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px
import warnings
warnings.simplefilter("ignore", FutureWarning)
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
ropls = importr('ropls')
base = importr('base')

def OPLSDA(data,metadata,y_var='timepoint_r',applylog=True):
    X = data.copy()
    X = X.loc[[i for i in X.index if i in metadata.index], :]
    if applylog:
        X = np.log2(X+1)
    # Y: response vector aligned to X
    Y = metadata.loc[X.index, y_var]  # adjust column name as needed
    
    # --- Convert to R objects ---
    with localconverter(ro.default_converter + pandas2ri.converter):
        X_r = ro.conversion.py2rpy(X)          # R matrix (samples x features)
        Y_r = ro.StrVector(Y.tolist())          # character vector for PLS-DA
    # --- Run OPLS-DA (1 predictive + orthogonal components) ---
    oplsda_model = ropls.opls(
        X_r,
        Y_r,
        predI  = 1,
        orthoI = ro.NA_Integer,   # auto-select number of orthogonal components
    )
    
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        # Scores (T scores = predictive, To = orthogonal)
        T_scores  = ro.conversion.rpy2py(ropls.getScoreMN(oplsda_model))
        To_scores = ro.conversion.rpy2py(ropls.getScoreMN(oplsda_model, orthoL=True))
    
        # Loadings
        P_load  = ro.conversion.rpy2py(ropls.getLoadingMN(oplsda_model))
        Po_load = ro.conversion.rpy2py(ropls.getLoadingMN(oplsda_model, orthoL=True))

        # VIP scores (predictive importance)
        vip = ro.conversion.rpy2py(ropls.getVipVn(oplsda_model))
        model_df = ro.conversion.rpy2py(oplsda_model.slots["modelDF"])
    T_df  = pd.DataFrame(T_scores,  index=X.index)
    To_df = pd.DataFrame(To_scores, index=X.index)
    vip_s = pd.Series(vip, index=X.columns, name='VIP')
    display = pd.concat([T_df, To_df.iloc[:, 0]], axis=1)
    display.columns=['Predictive',"Orthogonal"]
    display['sample_type'] = metadata.loc[display.index, y_var]
    sns.scatterplot(display,x='Predictive',y='Orthogonal',hue='sample_type')
    plt.title("OPLS-DA\n Predictive Component")
    return {"Predictive":T_df,"Orthogonal":To_df,"Predictive_Loadings":P_load,"Orthogonal_Loadings":Po_load,"VIP_Scores":vip_s,"Model_Statistics":model_df}
            
    

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


