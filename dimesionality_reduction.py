import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
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
