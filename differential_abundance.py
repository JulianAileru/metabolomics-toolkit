import contextlib
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import patsy
from scipy import stats
from inmoose.limma.squeezeVar import squeezeVar
from common_stats import calc_skew, calc_kurtosis
import warnings
warnings.simplefilter("ignore", FutureWarning)


class DifferentialAbundance:
    """
    Per-signal differential abundance testing for a two-group contrast.
    Default test ('ttest') fits OLS via the normal equations for all signals
    at once and supports covariates; an unadjusted Mann-Whitney U test
    ('mannwhitney') is also available per-feature for signals whose
    within-group distribution doesn't look normal enough for the t-test.
    A limma-style moderated t-test ('moderated_ttest') is also available,
    shrinking each feature's variance estimate toward a value borrowed
    across all features (via inmoose's empirical-Bayes squeezeVar) instead
    of trusting each feature's own variance in isolation — useful when n
    per group is small relative to the number of features.
    Pass paired=True (with a subject_id-style column in metadata) when the
    same subject contributes to both groups, e.g. two timepoints per
    individual — this switches to a paired t-test / Wilcoxon signed-rank
    test on within-subject differences instead of treating samples as
    independent, and drops any subject without exactly one sample per group.

    data : pd.DataFrame, shape (n_samples, n_features)
        Peak intensity matrix, samples x signals.
    metadata : pd.DataFrame, shape (n_samples, n_contrasts)
        Sample metadata indexed by sample, holding the grouping/contrast and
        covariate columns.
    """

    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self.results_ = {}
        self.stats_ = {}
        self.last_contrast_ = None

    @staticmethod
    def benjamini_hochberg(pvalues):
        """
        Benjamini-Hochberg FDR adjustment.

        pvalues : array-like of raw p-values. Entries that aren't a finite
            number in [0, 1] (e.g. NaN from a 0/0 t-stat on a zero-variance
            feature) are excluded from the correction and reported back as
            NaN, since `scipy.stats.false_discovery_control` rejects them
            outright.
        Returns an array of adjusted p-values in the same order as the input.
        """
        pvalues = np.asarray(pvalues, dtype=float)
        p_adj = np.full_like(pvalues, np.nan)
        valid = np.isfinite(pvalues) & (pvalues >= 0) & (pvalues <= 1)
        if valid.any():
            p_adj[valid] = stats.false_discovery_control(pvalues[valid], method='bh')
        return p_adj

    def _build_design_matrix(self, contrast, covariates=None):
        """
        Build an OLS design matrix for a two-group contrast plus optional
        covariates, using patsy formulas.

        contrast : tuple (column, level, reference)
            column    - metadata column holding the grouping variable
            level     - group of interest, coded 1 (numerator of the fold change)
            reference - baseline group, coded 0 (denominator of the fold change)
        covariates : list[str], optional
            Additional terms to adjust for, in patsy formula syntax. A plain
            column name lets patsy infer categorical (Treatment-coded,
            reference = first sorted level) vs continuous from its dtype.
            Any other patsy term is also accepted, e.g. 'C(batch, Sum)',
            'C(batch, Helmert)', 'bs(age, df=4)', or interactions like
            'batch:age', which is what makes alternate encodings possible.

        Returns the design matrix (samples x predictors, with an 'Intercept'
        column) and the name of the contrast column within it.
        """
        column, level, reference = contrast
        covariates = covariates or []
        metadata = self.metadata

        other_levels = set(metadata[column].dropna().unique()) - {level, reference}
        keep = metadata[column].isin([level, reference])
        if other_levels:
            print(f"Dropping {(~keep).sum()} sample(s) outside contrast levels {other_levels} for '{column}'")
        meta = metadata.loc[keep].copy()

        contrast_term = f"C({column}, Treatment(reference={reference!r}))"
        formula = " + ".join([contrast_term] + list(covariates))

        design = patsy.dmatrix(formula, data=meta, return_type='dataframe')
        dropped = meta.index.difference(design.index)
        for s in dropped:
            print(f"Dropping sample '{s}': missing contrast/covariate value")

        candidates = [c for c in design.columns if c.startswith(f"{contrast_term}[T.")]
        if len(candidates) != 1:
            raise ValueError(f"Could not isolate a single contrast column for {contrast} among {list(design.columns)}")
        contrast_name = candidates[0]

        return design, contrast_name

    @staticmethod
    def _ols_normal_equations(X, Y):
        """
        Fit OLS for many responses sharing one design matrix via the normal
        equations: beta = (X'X)^-1 X'Y.

        X : (n_samples, n_predictors) design matrix
        Y : (n_samples, n_signals) response matrix (e.g. log2 intensities)

        Returns beta (n_predictors, n_signals), (X'X)^-1, residual variance per
        signal (n_signals,), and the residual degrees of freedom.
        """
        n, p = X.shape
        if n <= p:
            raise ValueError(f"Not enough samples ({n}) to fit {p} predictors")
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ Y
        residuals = Y - X @ beta
        dof = n - p
        sigma2 = np.sum(residuals ** 2, axis=0) / dof
        return beta, XtX_inv, sigma2, dof

    def fit(self, contrast, covariates=None, applylog=True, pseudocount=1, test='ttest',
            paired=False, subject_col='subject_id', equal_var=True):
        """
        Run the differential abundance test for one contrast.

        contrast : tuple (column, level, reference)
            Two-group comparison to test, e.g. ('sample_type', 'Case', 'Control').
            log2FC/FC is level vs reference.
        covariates : list[str], optional
            Patsy formula terms to adjust for (see `_build_design_matrix`).
            Only supported when paired=False, test='ttest', equal_var=True —
            Mann-Whitney, the paired tests, and Welch's t-test (equal_var=False)
            don't support covariate adjustment.
        applylog : bool, default True
            log2-transform the data before fitting, so log2FC and the test
            are computed on the same scale. If False, the effect size is
            reported as a linear fold change ('FC' = ratio of group
            means/medians, computed directly from the two groups' raw
            values) instead of 'log2FC'. Not supported together with
            `covariates` for test='ttest' (equal_var=True) or
            test='moderated_ttest', since a ratio doesn't compose
            additively with covariate adjustment the way a log-scale
            difference does — use applylog=True if you need a
            covariate-adjusted effect size.
        pseudocount : float, default 1
            Added before log2 transform to avoid log(0), and added to
            numerator/denominator when applylog=False to avoid dividing by
            zero in the FC ratio.
        test : 'ttest', 'mannwhitney', or 'moderated_ttest', default 'ttest'
            Which family of test to run — parametric or rank-based — for
            either independent or paired samples (see `paired`).
            'ttest': independent two-sample t-test (OLS-based, supports
            covariates, when equal_var=True; otherwise Welch's t-test with
            per-group variances), or, if paired, a paired t-test on
            per-subject differences.
            'mannwhitney': independent Mann-Whitney U test, or, if paired, a
            Wilcoxon signed-rank test on per-subject differences. Useful for
            features `calculate_stats`/`flag_nonparametric_candidates` flag
            as too skewed/heavy-tailed or too small-n for the t-test.
            'moderated_ttest': limma-style moderated t-test (independent
            samples only, paired=False). Same OLS fit as 'ttest', but each
            feature's residual variance is shrunk toward a common prior
            estimated empirically across all features (via inmoose's port
            of limma's squeezeVar/fitFDist), and tested against inflated
            degrees of freedom. Stabilizes variance estimates — and so
            p-values — for small sample sizes, at the cost of assuming most
            features share similar variance. Supports covariates.
        paired : bool, default False
            Set True when the same subject contributes a sample to both
            groups (e.g. two timepoints per individual), so the test should
            account for within-subject correlation instead of treating all
            samples as independent. Requires `subject_col` in metadata;
            subjects without exactly one sample in each group are dropped
            (see `_pair_samples`).
        subject_col : str, default 'subject_id'
            Metadata column identifying which samples belong to the same
            subject. Only used when paired=True.
        equal_var : bool, default True
            Only used when test='ttest', paired=False. If True, pools a
            single residual variance across both groups (OLS, supports
            covariates). If False, runs Welch's t-test instead — per-group
            variance and per-feature Welch-Satterthwaite degrees of freedom,
            no pooling assumption. Does not support covariates.

        Returns a DataFrame indexed by signal with columns log2FC (or FC if
        applylog=False), p_value, p_adj (plus t_stat/U_stat/W_stat depending
        on test), sorted by p_value. Also cached on `self.results_` keyed by
        the contrast name; pass that DataFrame to `volcano_plot`.
        """
        if paired:
            if covariates:
                raise ValueError("covariates are not supported with paired=True")
            if test == 'ttest':
                results, contrast_name = self._fit_paired_ttest(contrast, applylog, pseudocount, subject_col)
            elif test == 'mannwhitney':
                results, contrast_name = self._fit_wilcoxon(contrast, applylog, pseudocount, subject_col)
            elif test == 'moderated_ttest':
                raise ValueError("moderated_ttest does not support paired=True; use test='ttest' for paired data")
            else:
                raise ValueError(f"Unknown test '{test}'; expected 'ttest', 'mannwhitney', or 'moderated_ttest'")
        else:
            if test == 'ttest':
                results, contrast_name = self._fit_ttest(contrast, covariates, applylog, pseudocount, equal_var)
            elif test == 'mannwhitney':
                if covariates:
                    raise ValueError("covariates are not supported with test='mannwhitney' (unadjusted two-sample rank test)")
                results, contrast_name = self._fit_mannwhitney(contrast, applylog, pseudocount)
            elif test == 'moderated_ttest':
                results, contrast_name = self._fit_moderated_ttest(contrast, covariates, applylog, pseudocount)
            else:
                raise ValueError(f"Unknown test '{test}'; expected 'ttest', 'mannwhitney', or 'moderated_ttest'")

        self.results_[contrast_name] = results
        self.last_contrast_ = contrast_name
        return results

    def _pair_samples(self, contrast, subject_col):
        """
        Resolve a contrast to matched (level, reference) sample pairs sharing
        a subject. Drops any sample outside the contrast's two levels, any
        sample missing from `self.data`, and any subject that does not have
        exactly one sample in each group (no pair, or duplicate measurements
        in one group), printing the reason for each drop.

        Returns parallel lists `level_samples`, `ref_samples`, ordered by
        subject so that index i in each refers to the same subject.
        """
        column, level, reference = contrast
        metadata = self.metadata

        other_levels = set(metadata[column].dropna().unique()) - {level, reference}
        keep = metadata[column].isin([level, reference])
        if other_levels:
            print(f"Dropping {(~keep).sum()} sample(s) outside contrast levels {other_levels} for '{column}'")
        meta = metadata.loc[keep]

        common = self.data.index.intersection(meta.index)
        missing = meta.index.difference(self.data.index)
        for s in missing:
            print(f"Dropping sample '{s}': not present in data")
        meta = meta.loc[common]

        level_samples, ref_samples = [], []
        for subject, group_df in meta.groupby(subject_col):
            subject_level = group_df.index[group_df[column] == level]
            subject_ref = group_df.index[group_df[column] == reference]
            if len(subject_level) != 1 or len(subject_ref) != 1:
                print(f"Dropping subject '{subject}': expected exactly one {level} and one {reference} "
                      f"sample, found {len(subject_level)} and {len(subject_ref)}")
                continue
            level_samples.append(subject_level[0])
            ref_samples.append(subject_ref[0])

        return level_samples, ref_samples

    def _fit_ttest(self, contrast, covariates, applylog, pseudocount, equal_var=True):
        if not equal_var:
            if covariates:
                raise ValueError("covariates are not supported with equal_var=False (Welch's t-test)")
            return self._fit_welch_ttest(contrast, applylog, pseudocount)

        if not applylog and covariates:
            raise ValueError("linear fold change (applylog=False) is not supported with covariates; "
                              "a ratio doesn't compose additively with covariate adjustment — use applylog=True")

        design, contrast_col = self._build_design_matrix(contrast, covariates)

        common = self.data.index.intersection(design.index)
        missing = design.index.difference(self.data.index)
        for s in missing:
            print(f"Dropping sample '{s}': not present in data")
        design = design.loc[common]
        Y = self.data.loc[common].apply(pd.to_numeric)

        if applylog:
            Y = np.log2(Y + pseudocount)

        X = design.values
        beta, XtX_inv, sigma2, dof = self._ols_normal_equations(X, Y.values)

        contrast_idx = design.columns.get_loc(contrast_col)
        se = np.sqrt(sigma2 * XtX_inv[contrast_idx, contrast_idx])
        effect = beta[contrast_idx]
        t_stat = effect / se
        p_value = 2 * stats.t.sf(np.abs(t_stat), df=dof)

        is_level = design[contrast_col] == 1
        if applylog:
            fc_col, fc_value = 'log2FC', effect
        else:
            mean_level = Y.loc[is_level].mean(axis=0)
            mean_ref = Y.loc[~is_level].mean(axis=0)
            fc_col, fc_value = 'FC', (mean_level + pseudocount) / (mean_ref + pseudocount)

        results = pd.DataFrame({
            fc_col: fc_value,
            't_stat': t_stat,
            'p_value': p_value,
        }, index=Y.columns)
        results['p_adj'] = self.benjamini_hochberg(results['p_value'].values)
        results = self._merge_descriptive_stats(results, contrast, applylog, pseudocount)
        results = results.sort_values('p_value')

        column, level, reference = contrast
        contrast_name = f"{column}[{level} vs {reference}]"
        n_level = int((design[contrast_col] == 1).sum())
        n_ref = int((design[contrast_col] == 0).sum())
        print(f"Contrast: {contrast_name}  ({level}: n={n_level}, {reference}: n={n_ref})  test=ttest  covariates={covariates or []}  dof={dof}")

        return results, contrast_name

    def _fit_moderated_ttest(self, contrast, covariates, applylog, pseudocount):
        """
        limma-style moderated t-test: same per-feature OLS fit as
        `_fit_ttest`, but `sigma2`/`dof` are passed to inmoose's `squeezeVar`
        (a port of limma's squeezeVar/fitFDist) to shrink each feature's
        residual variance toward a common empirical-Bayes prior estimated
        across all features, rather than trusting each feature's own
        variance estimate in isolation. The moderated t-stat and inflated
        degrees of freedom follow limma's eBayes formula directly; inmoose's
        own eBayes() is not used here because its B-statistic step raises a
        KeyError whenever df_prior is estimated as infinite (a real,
        legitimate squeezeVar result, not a rare corner case) — squeezeVar
        itself handles that case correctly.
        """
        if not applylog and covariates:
            raise ValueError("linear fold change (applylog=False) is not supported with covariates; "
                              "a ratio doesn't compose additively with covariate adjustment — use applylog=True")

        design, contrast_col = self._build_design_matrix(contrast, covariates)

        common = self.data.index.intersection(design.index)
        missing = design.index.difference(self.data.index)
        for s in missing:
            print(f"Dropping sample '{s}': not present in data")
        design = design.loc[common]
        Y = self.data.loc[common].apply(pd.to_numeric)

        if applylog:
            Y = np.log2(Y + pseudocount)

        X = design.values
        beta, XtX_inv, sigma2, dof = self._ols_normal_equations(X, Y.values)

        n_features = sigma2.shape[0]
        squeezed = squeezeVar(sigma2, dof)
        s2_post = squeezed['var_post']
        df_prior = squeezed['df_prior']
        df_pooled = dof * n_features
        df_total = np.minimum(dof + df_prior, df_pooled)

        contrast_idx = design.columns.get_loc(contrast_col)
        se = np.sqrt(s2_post * XtX_inv[contrast_idx, contrast_idx])
        effect = beta[contrast_idx]
        t_stat = effect / se
        p_value = 2 * stats.t.sf(np.abs(t_stat), df=df_total)

        is_level = design[contrast_col] == 1
        if applylog:
            fc_col, fc_value = 'log2FC', effect
        else:
            mean_level = Y.loc[is_level].mean(axis=0)
            mean_ref = Y.loc[~is_level].mean(axis=0)
            fc_col, fc_value = 'FC', (mean_level + pseudocount) / (mean_ref + pseudocount)

        results = pd.DataFrame({
            fc_col: fc_value,
            't_stat': t_stat,
            'p_value': p_value,
        }, index=Y.columns)
        results['p_adj'] = self.benjamini_hochberg(results['p_value'].values)
        results = self._merge_descriptive_stats(results, contrast, applylog, pseudocount)
        results = results.sort_values('p_value')

        column, level, reference = contrast
        contrast_name = f"{column}[{level} vs {reference}]"
        n_level = int((design[contrast_col] == 1).sum())
        n_ref = int((design[contrast_col] == 0).sum())
        print(f"Contrast: {contrast_name}  ({level}: n={n_level}, {reference}: n={n_ref})  test=moderated_ttest  "
              f"covariates={covariates or []}  dof_residual={dof}  df_prior={df_prior:.3g}")

        return results, contrast_name

    def _fit_welch_ttest(self, contrast, applylog, pseudocount):
        """
        Independent two-sample t-test without pooling variance across groups:
        each group's variance is estimated separately and degrees of freedom
        are computed per-feature via the Welch-Satterthwaite equation, since
        they depend on each signal's own group variances.
        """
        column, level, reference = contrast
        metadata = self.metadata

        other_levels = set(metadata[column].dropna().unique()) - {level, reference}
        keep = metadata[column].isin([level, reference])
        if other_levels:
            print(f"Dropping {(~keep).sum()} sample(s) outside contrast levels {other_levels} for '{column}'")
        meta = metadata.loc[keep]

        common = self.data.index.intersection(meta.index)
        missing = meta.index.difference(self.data.index)
        for s in missing:
            print(f"Dropping sample '{s}': not present in data")

        Y = self.data.loc[common].apply(pd.to_numeric)
        if applylog:
            Y = np.log2(Y + pseudocount)

        level_samples = meta.index[meta[column] == level].intersection(common)
        ref_samples = meta.index[meta[column] == reference].intersection(common)

        level_df = Y.loc[level_samples]
        ref_df = Y.loc[ref_samples]

        n_level = level_df.notna().sum(axis=0)
        n_ref = ref_df.notna().sum(axis=0)
        mean_level = level_df.mean(axis=0)
        mean_ref = ref_df.mean(axis=0)
        var_level = level_df.var(axis=0, ddof=1)
        var_ref = ref_df.var(axis=0, ddof=1)

        se2_level = var_level / n_level
        se2_ref = var_ref / n_ref
        se = np.sqrt(se2_level + se2_ref)
        effect = mean_level - mean_ref
        t_stat = effect / se
        dof = (se2_level + se2_ref) ** 2 / (
            se2_level ** 2 / (n_level - 1) + se2_ref ** 2 / (n_ref - 1)
        )
        p_value = 2 * stats.t.sf(np.abs(t_stat), df=dof)

        if applylog:
            fc_col, fc_value = 'log2FC', effect
        else:
            fc_col, fc_value = 'FC', (mean_level + pseudocount) / (mean_ref + pseudocount)

        results = pd.DataFrame({
            fc_col: fc_value,
            't_stat': t_stat,
            'p_value': p_value,
        }, index=Y.columns)
        results['p_adj'] = self.benjamini_hochberg(results['p_value'].values)
        results = self._merge_descriptive_stats(results, contrast, applylog, pseudocount)
        results = results.sort_values('p_value')

        contrast_name = f"{column}[{level} vs {reference}]"
        print(f"Contrast: {contrast_name}  ({level}: n={len(level_samples)}, {reference}: n={len(ref_samples)})  "
              f"test=ttest  equal_var=False (Welch)  dof=per-feature")

        return results, contrast_name

    def _fit_mannwhitney(self, contrast, applylog, pseudocount):
        column, level, reference = contrast
        metadata = self.metadata

        other_levels = set(metadata[column].dropna().unique()) - {level, reference}
        keep = metadata[column].isin([level, reference])
        if other_levels:
            print(f"Dropping {(~keep).sum()} sample(s) outside contrast levels {other_levels} for '{column}'")
        meta = metadata.loc[keep]

        common = self.data.index.intersection(meta.index)
        missing = meta.index.difference(self.data.index)
        for s in missing:
            print(f"Dropping sample '{s}': not present in data")

        Y = self.data.loc[common].apply(pd.to_numeric)
        if applylog:
            Y = np.log2(Y + pseudocount)

        level_samples = meta.index[meta[column] == level].intersection(common)
        ref_samples = meta.index[meta[column] == reference].intersection(common)

        level_vals = Y.loc[level_samples].values
        ref_vals = Y.loc[ref_samples].values

        u_stat, p_value = stats.mannwhitneyu(level_vals, ref_vals, axis=0, alternative='two-sided', nan_policy='omit')
        median_level = np.median(level_vals, axis=0)
        median_ref = np.median(ref_vals, axis=0)
        if applylog:
            fc_col, fc_value = 'log2FC', median_level - median_ref
        else:
            fc_col, fc_value = 'FC', (median_level + pseudocount) / (median_ref + pseudocount)

        results = pd.DataFrame({
            fc_col: fc_value,
            'U_stat': u_stat,
            'p_value': p_value,
        }, index=Y.columns)
        results['p_adj'] = self.benjamini_hochberg(results['p_value'].values)
        results = self._merge_descriptive_stats(results, contrast, applylog, pseudocount)
        results = results.sort_values('p_value')

        contrast_name = f"{column}[{level} vs {reference}]"
        print(f"Contrast: {contrast_name}  ({level}: n={len(level_samples)}, {reference}: n={len(ref_samples)})  test=mannwhitney")

        return results, contrast_name

    def _fit_paired_ttest(self, contrast, applylog, pseudocount, subject_col):
        column, level, reference = contrast
        level_samples, ref_samples = self._pair_samples(contrast, subject_col)
        n = len(level_samples)
        if n < 2:
            raise ValueError(f"Not enough paired subjects ({n}) for a paired t-test")

        Y_level = self.data.loc[level_samples].apply(pd.to_numeric)
        Y_ref = self.data.loc[ref_samples].apply(pd.to_numeric)
        if applylog:
            Y_level = np.log2(Y_level + pseudocount)
            Y_ref = np.log2(Y_ref + pseudocount)

        diff = Y_level.values - Y_ref.values
        mean_diff = diff.mean(axis=0)
        se = diff.std(axis=0, ddof=1) / np.sqrt(n)
        t_stat = mean_diff / se
        dof = n - 1
        p_value = 2 * stats.t.sf(np.abs(t_stat), df=dof)

        if applylog:
            fc_col, fc_value = 'log2FC', mean_diff
        else:
            mean_level = Y_level.mean(axis=0)
            mean_ref = Y_ref.mean(axis=0)
            fc_col, fc_value = 'FC', (mean_level + pseudocount) / (mean_ref + pseudocount)

        results = pd.DataFrame({
            fc_col: fc_value,
            't_stat': t_stat,
            'p_value': p_value,
        }, index=self.data.columns)
        results['p_adj'] = self.benjamini_hochberg(results['p_value'].values)
        results = self._merge_descriptive_stats(results, contrast, applylog, pseudocount, paired=True, subject_col=subject_col)
        results = results.sort_values('p_value')

        contrast_name = f"{column}[{level} vs {reference}] (paired)"
        print(f"Contrast: {contrast_name}  (n_pairs={n})  test=ttest  paired=True  dof={dof}")

        return results, contrast_name

    def _fit_wilcoxon(self, contrast, applylog, pseudocount, subject_col):
        column, level, reference = contrast
        level_samples, ref_samples = self._pair_samples(contrast, subject_col)
        n = len(level_samples)
        if n < 1:
            raise ValueError(f"Not enough paired subjects ({n}) for a Wilcoxon signed-rank test")

        Y_level = self.data.loc[level_samples].apply(pd.to_numeric)
        Y_ref = self.data.loc[ref_samples].apply(pd.to_numeric)
        if applylog:
            Y_level = np.log2(Y_level + pseudocount)
            Y_ref = np.log2(Y_ref + pseudocount)

        w_stat, p_value = stats.wilcoxon(Y_level.values, Y_ref.values, axis=0,
                                          zero_method='wilcox', alternative='two-sided')
        if applylog:
            fc_col, fc_value = 'log2FC', np.median(Y_level.values - Y_ref.values, axis=0)
        else:
            median_level = np.median(Y_level.values, axis=0)
            median_ref = np.median(Y_ref.values, axis=0)
            fc_col, fc_value = 'FC', (median_level + pseudocount) / (median_ref + pseudocount)

        results = pd.DataFrame({
            fc_col: fc_value,
            'W_stat': w_stat,
            'p_value': p_value,
        }, index=self.data.columns)
        results['p_adj'] = self.benjamini_hochberg(results['p_value'].values)
        results = self._merge_descriptive_stats(results, contrast, applylog, pseudocount, paired=True, subject_col=subject_col)
        results = results.sort_values('p_value')

        contrast_name = f"{column}[{level} vs {reference}] (paired)"
        print(f"Contrast: {contrast_name}  (n_pairs={n})  test=mannwhitney(wilcoxon)  paired=True")

        return results, contrast_name

    def _merge_descriptive_stats(self, results, contrast, applylog, pseudocount, paired=False, subject_col='subject_id'):
        """
        Join mean/median/std/n descriptive stats from `calculate_stats` onto
        a results table — per-group (level, reference) for independent
        tests, or per-subject-difference for paired tests, matching whatever
        `calculate_stats` reports for that `paired` setting. Skew/kurtosis
        are left out since they're for the nonparametric-candidate screen,
        not for reading alongside a test result.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            stats_table = self.calculate_stats(contrast, applylog=applylog, pseudocount=pseudocount,
                                                paired=paired, subject_col=subject_col)
        if paired:
            cols = ['diff_n', 'diff_mean', 'diff_median', 'diff_std']
        else:
            _, level, reference = contrast
            cols = [f'{level}_n', f'{level}_mean', f'{level}_median', f'{level}_std',
                    f'{reference}_n', f'{reference}_mean', f'{reference}_median', f'{reference}_std']
        return results.join(stats_table[cols])

    def calculate_stats(self, contrast, applylog=True, pseudocount=1, paired=False, subject_col='subject_id'):
        """
        Per-feature descriptive statistics, for sanity-checking an extreme
        log2FC (e.g. driven by a couple of outliers vs. a genuine shift) and
        for judging whether a nonparametric test would be more appropriate
        than the OLS/paired t-test in `fit`.

        contrast : tuple (column, level, reference)
            Same two-group contrast spec used by `fit`.
        applylog : bool, default True
            log2-transform before computing stats, matching the scale `fit`
            reports log2FC on.
        pseudocount : float, default 1
            Added before log2 transform to avoid log(0).
        paired : bool, default False
            If False (default), returns per-group stats: mean, median, std,
            skew, kurtosis of each of `level` and `reference` separately —
            relevant for the independent-samples tests.
            If True, returns stats of the per-subject `level - reference`
            difference instead (mean, median, std, skew, kurtosis, n) —
            this is what actually determines whether a paired t-test or a
            Wilcoxon signed-rank test is appropriate, since the paired tests
            care about the distribution of the differences, not of the raw
            groups. Uses `_pair_samples` so unpaired subjects are dropped
            exactly as in `fit(..., paired=True)`.
        subject_col : str, default 'subject_id'
            Metadata column identifying matched subjects. Only used when paired=True.

        Returns a DataFrame indexed by signal. Unpaired columns:
        {level}_n, {level}_mean, {level}_median, {level}_std, {level}_skew,
        {level}_kurtosis, and the same for `reference` (`n` is the
        per-feature non-missing sample count within the group, since
        untargeted data often has per-feature dropouts). Paired columns:
        diff_n, diff_mean, diff_median, diff_std, diff_skew, diff_kurtosis.
        Cached on `self.stats_`, keyed the same way as `self.results_`.
        """
        column, level, reference = contrast

        if paired:
            level_samples, ref_samples = self._pair_samples(contrast, subject_col)
            Y_level = self.data.loc[level_samples].apply(pd.to_numeric)
            Y_ref = self.data.loc[ref_samples].apply(pd.to_numeric)
            if applylog:
                Y_level = np.log2(Y_level + pseudocount)
                Y_ref = np.log2(Y_ref + pseudocount)
            diff = pd.DataFrame(Y_level.values - Y_ref.values, columns=self.data.columns, index=Y_level.index)

            stats_table = pd.DataFrame({
                'diff_n': diff.notna().sum(axis=0),
                'diff_mean': diff.mean(axis=0),
                'diff_median': diff.median(axis=0),
                'diff_std': diff.std(axis=0, ddof=1),
                'diff_skew': calc_skew(diff, qc=''),
                'diff_kurtosis': calc_kurtosis(diff, qc=''),
            })
            contrast_name = f"{column}[{level} vs {reference}] (paired)"
        else:
            metadata = self.metadata
            keep = metadata[column].isin([level, reference])
            meta = metadata.loc[keep]
            common = self.data.index.intersection(meta.index)
            missing = meta.index.difference(self.data.index)
            for s in missing:
                print(f"Dropping sample '{s}': not present in data")

            Y = self.data.loc[common].apply(pd.to_numeric)
            if applylog:
                Y = np.log2(Y + pseudocount)

            group_tables = {}
            for group_label in (level, reference):
                group_samples = meta.index[meta[column] == group_label].intersection(common)
                group_data = Y.loc[group_samples]
                group_tables[group_label] = pd.DataFrame({
                    'n': group_data.notna().sum(axis=0),
                    'mean': group_data.mean(axis=0),
                    'median': group_data.median(axis=0),
                    'std': group_data.std(axis=0, ddof=1),
                    'skew': calc_skew(group_data, qc=''),
                    'kurtosis': calc_kurtosis(group_data, qc=''),
                })

            stats_table = pd.concat(group_tables, axis=1)
            stats_table.columns = [f"{group}_{stat}" for group, stat in stats_table.columns]
            contrast_name = f"{column}[{level} vs {reference}]"

        self.stats_[contrast_name] = stats_table
        return stats_table

    def flag_nonparametric_candidates(self, contrast, skew_thresh=2.0, kurt_thresh=2.0, min_n=8,
                                       paired=False, subject_col='subject_id'):
        """
        Heuristic screen over `calculate_stats` for features whose relevant
        distribution looks non-normal enough, or whose sample size is small
        enough, that the t-test in `fit` may be unreliable and a
        nonparametric test (`fit(..., test='mannwhitney')`) is worth trying.
        This is advisory only — it does not change which test `fit` runs.

        contrast : tuple (column, level, reference)
        skew_thresh, kurt_thresh : float
            Flag a feature if |skew| or |kurtosis| exceeds this.
            Unpaired: checked in either group. Paired: checked on the
            per-subject differences, which is what actually matters for
            choosing paired-t vs. Wilcoxon.
        min_n : int
            Flag a feature if its (group, or paired-diff) sample size is
            smaller than this.
        paired, subject_col :
            Forwarded to `calculate_stats` — see there for details. Must
            match how you intend to call `fit`.

        Returns the list of flagged feature names.
        """
        column, level, reference = contrast
        contrast_name = f"{column}[{level} vs {reference}]" + (" (paired)" if paired else "")
        stats_table = self.stats_.get(contrast_name)
        if stats_table is None:
            stats_table = self.calculate_stats(contrast, paired=paired, subject_col=subject_col)

        if paired:
            flagged = (
                (stats_table['diff_skew'].abs() > skew_thresh) |
                (stats_table['diff_kurtosis'].abs() > kurt_thresh) |
                (stats_table['diff_n'] < min_n)
            )
        else:
            flagged = pd.Series(False, index=stats_table.index)
            for group_label in (level, reference):
                flagged |= stats_table[f'{group_label}_skew'].abs() > skew_thresh
                flagged |= stats_table[f'{group_label}_kurtosis'].abs() > kurt_thresh
                flagged |= stats_table[f'{group_label}_n'] < min_n

        return stats_table.index[flagged].tolist()

    @staticmethod
    def volcano_plot(results, fc_thresh=1.0, p_thresh=0.05, use_adjusted=True, top_n=10,
                      title=None, output_file=None, backend='seaborn'):
        """
        Standard differential abundance volcano plot: -log10(p-value) vs the
        effect size column in `results`, either 'log2FC' or, if `fit` was
        run with applylog=False, the linear fold change 'FC'.

        results : pd.DataFrame
            Output of `fit`.
        fc_thresh : float
            For 'log2FC' results: |log2FC| cutoff for calling a signal
            up/down (additive, symmetric around 0). For 'FC' results:
            multiplicative cutoff — up if FC >= fc_thresh, down if
            FC <= 1/fc_thresh (symmetric around 1 on a ratio scale). The
            default of 1.0 suits log2FC out of the box; for FC results pass
            a value > 1 (e.g. 2.0 for a two-fold cutoff).
        p_thresh : float
            Significance cutoff, applied to p_adj if use_adjusted else p_value.
        use_adjusted : bool
            Whether the significance cutoff applies to p_adj or raw p_value.
        top_n : int
            Number of top hits (by p_value) to label on a static (seaborn) plot.
        backend : 'seaborn' or 'plotly'
        """
        df = results.copy()
        fc_col = 'log2FC' if 'log2FC' in df.columns else 'FC'
        smallest_nonzero = df.loc[df['p_value'] > 0, 'p_value'].min() if (df['p_value'] > 0).any() else np.nextafter(0, 1)
        df['neg_log10_p'] = -np.log10(df['p_value'].replace(0, smallest_nonzero))

        sig_col = 'p_adj' if use_adjusted else 'p_value'
        down_thresh = 1 / fc_thresh if fc_col == 'FC' else -fc_thresh

        def classify(row):
            if row[sig_col] >= p_thresh:
                return 'Not significant'
            if row[fc_col] >= fc_thresh:
                return 'Up'
            if row[fc_col] <= down_thresh:
                return 'Down'
            return 'Not significant'

        df['direction'] = df.apply(classify, axis=1)
        colors = {'Up': '#d62728', 'Down': '#1f77b4', 'Not significant': '#7f7f7f'}
        plot_title = title or 'Differential Abundance'

        if backend == 'plotly':
            df['feature'] = df.index
            fig = px.scatter(
                df, x=fc_col, y='neg_log10_p', color='direction',
                color_discrete_map=colors, hover_data=['feature'], title=plot_title,
                labels={'neg_log10_p': '-log10(p-value)'},
            )
            fig.add_vline(x=fc_thresh, line_dash='dash', line_color='grey')
            fig.add_vline(x=down_thresh, line_dash='dash', line_color='grey')
            fig.add_hline(y=-np.log10(p_thresh), line_dash='dash', line_color='grey')
            fig.show()
            if output_file:
                fig.write_html(output_file)
        else:
            plt.figure()
            for direction, color in colors.items():
                subset = df[df['direction'] == direction]
                plt.scatter(subset[fc_col], subset['neg_log10_p'], c=color, label=direction, s=15, alpha=0.7)
            plt.axvline(fc_thresh, ls='--', c='grey')
            plt.axvline(down_thresh, ls='--', c='grey')
            plt.axhline(-np.log10(p_thresh), ls='--', c='grey')
            plt.xlabel(fc_col)
            plt.ylabel('-log10(p-value)')
            plt.title(plot_title)
            plt.legend()
            top_hits = df.sort_values('p_value').head(top_n)
            for idx, row in top_hits.iterrows():
                plt.annotate(str(idx), (row[fc_col], row['neg_log10_p']), fontsize=7)
            plt.tight_layout()
            if output_file:
                plt.savefig(output_file, bbox_inches='tight')

        return df
