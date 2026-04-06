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
import os
r_bin = r'C:\Users\jaileru\AppData\Local\miniconda3\envs\home_env\Lib\R\bin\x64'
os.environ['PATH'] = r_bin + os.pathsep + os.environ['PATH']
ropls = importr('ropls')
base = importr('base')
biobase = importr("Biobase")
pvca_pkg = importr("pvca")

def pca_plot(data, metadata, hue=['timepoint', 'sample_type', 'instrument'], title=None,output_file=None,ignore_blanks=True,applylog=True):
    """
    For each hue in the list, selects samples present in both data and metadata
    (with non-null values for that column), runs PCA, and plots a scatterplot.

    hue options: 'timepoint' (uses 'timepoint_r'), 'sample_type', 'instrument'
    """
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    if ignore_blanks:
        data = data[~data.index.str.contains("_B_")]
    col_map = {
        'timepoint': 'timepoint_r',
        'sample_type': 'sample_type',
        'instrument': 'instrument',
    }

    for i,h in enumerate(hue):
        col = col_map.get(h, h)
        if col not in metadata.columns:
            raise ValueError(f"Column '{col}' not found in metadata")

        # Select samples present in both data index and metadata, with non-null hue values
        aligned_meta = metadata.reindex(data.index).dropna(subset=[col])
        filtered_data = data.loc[aligned_meta.index]

        if filtered_data.empty:
            print(f"No samples found for hue='{h}', skipping.")
            continue

        pca_coords = pca.fit_transform(scaler.fit_transform(np.log2(filtered_data + 1) if applylog else filtered_data))
        pca_df = pd.DataFrame(pca_coords, columns=['PC1', 'PC2'], index=filtered_data.index)
        pca_df[col] = aligned_meta[col]

        var1, var2 = pca.explained_variance_ratio_ * 100
        n_samples, n_signals = filtered_data.shape
        plt.figure()
        if title:
            plt.title(f"{title} — colored by {h}\nSignals:{n_signals} Samples:{n_samples}")
        else:
             plt.title(f'{h}\nSignals:{n_signals} Samples:{n_samples}')
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=col)
        plt.xlabel(f"PC1 ({var1:.1f}%)")
        plt.ylabel(f"PC2 ({var2:.1f}%)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if output_file and i < len(output_file):
            plt.savefig(output_file[i], bbox_inches='tight')
    return pca_df
        


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

def run_pvca(
    data: pd.DataFrame,
    sample_info: pd.DataFrame,
    batch_factors: list[str],
    threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Run PVCA on a feature matrix and return variance proportions per factor.
 
    Parameters
    ----------
    data : pd.DataFrame
        Feature matrix with shape (n_samples, n_features).
        Rows = samples, columns = features (metabolites, genes, etc.).
    sample_info : pd.DataFrame
        Sample metadata with shape (n_samples, n_factors).
        Must contain columns named in `batch_factors`.
        Index should match `data.index`.
    batch_factors : list[str]
        Column names in `sample_info` to include as sources of variation.
        Two-way interactions between all pairs are automatically included
        by PVCA.
    threshold : float, default 0.6
        Minimum cumulative proportion of variance that the selected
        principal components must explain (0 < threshold <= 1).
 
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["source", "variance_proportion"], sorted
        descending by variance proportion. Sources include main effects,
        pairwise interactions, and "resid" (residual).
 
    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> # Simulated metabolomics matrix: 40 samples x 200 metabolites
    >>> np.random.seed(42)
    >>> n_samples, n_features = 40, 200
    >>> data = pd.DataFrame(
    ...     np.random.randn(n_samples, n_features),
    ...     index=[f"S{i}" for i in range(n_samples)],
    ...     columns=[f"met_{i}" for i in range(n_features)],
    ... )
    >>> sample_info = pd.DataFrame({
    ...     "batch": np.repeat(["B1", "B2"], n_samples // 2),
    ...     "treatment": np.tile(["ctrl", "treat"], n_samples // 2),
    ... }, index=data.index)
    >>> result = run_pvca(data, sample_info, ["batch", "treatment"], threshold=0.6)
    >>> print(result)
    """

    # ── Validate inputs ──────────────────────────────────────────────
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
 
    missing = set(batch_factors) - set(sample_info.columns)
    if missing:
        raise ValueError(
            f"batch_factors {missing} not found in sample_info columns"
        )
 
    if not data.index.equals(sample_info.index):
        raise ValueError(
            "data.index and sample_info.index must match (same samples, same order)"
        )
 
    # ── Build the ExpressionSet in R ─────────────────────────────────
    # pvca expects an ExpressionSet where:
    #   - assayData: features x samples matrix (note: transposed from our input)
    #   - phenoData: sample metadata as an AnnotatedDataFrame
 
    # Convert feature matrix to R (features x samples)
    expr_matrix = ro.r["matrix"](
        ro.FloatVector(data.values.T.flatten()),
        nrow=data.shape[1],
        ncol=data.shape[0],
    )
    # Set dimnames
    ro.r.assign("expr_mat", expr_matrix)
    ro.r(
        f'rownames(expr_mat) <- c({",".join(repr(c) for c in data.columns)})'
    )
    ro.r(
        f'colnames(expr_mat) <- c({",".join(repr(s) for s in data.index)})'
    )
 
    # Build phenoData
    pheno_dict = {}
    for col in sample_info.columns:
        vals = sample_info[col].astype(str).tolist()
        pheno_dict[col] = ro.StrVector(vals)
 
    r_df = ro.DataFrame(pheno_dict)
    ro.r.assign("pheno_df", r_df)
    ro.r(
        f'rownames(pheno_df) <- c({",".join(repr(s) for s in sample_info.index)})'
    )
 
    # Create AnnotatedDataFrame and ExpressionSet
    ro.r(
        """
        pheno_ad <- new("AnnotatedDataFrame", data = pheno_df)
        eset <- ExpressionSet(assayData = expr_mat, phenoData = pheno_ad)
        """
    )
 
    # ── Run PVCA ─────────────────────────────────────────────────────
    batch_factors_r = ro.StrVector(batch_factors)
    ro.r.assign("batch_factors", batch_factors_r)
    ro.r.assign("pct_threshold", threshold)
 
    ro.r(
        """
        pvca_result <- pvcaBatchAssess(eset, batch.factors = batch_factors, threshold = pct_threshold)
        """
    )
 
    # ── Extract results ──────────────────────────────────────────────
    labels = list(ro.r("pvca_result$label"))
    values = list(ro.r("pvca_result$dat"))
 
    result = (
        pd.DataFrame({"source": labels, "variance_proportion": values})
        .sort_values("variance_proportion", ascending=False)
        .reset_index(drop=True)
    )
 
    return result
