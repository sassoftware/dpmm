import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from scipy import integrate
from scipy.stats import chi2
from scipy.optimize import root_scalar
from scipy.stats import norm, binomtest
from scipy.stats import beta as beta_dist

from riskcal.analysis import get_beta_from_gdp


def mu_lower_from_two_groups(y_D, y_Dp, alpha=0.1, sens=1.0):
    """
    y_D: array of releases on dataset D (same query, rerun mechanism many times)
    y_Dp: array of releases on dataset D' (neighboring dataset)
    alpha: one-sided error; returns mu_lo with confidence ~1-alpha
    sens: query sensitivity Δ (1.0 in our case)

    Model: Y = q(D) + N(0, sigma^2) and Y' = q(D') + N(0, sigma^2)
    mu = Δ / sigma
    """
    y_D = np.asarray(y_D, dtype=float)
    y_Dp = np.asarray(y_Dp, dtype=float)

    m = len(y_D)
    n = len(y_Dp)
    if m < 2 or n < 2:
        raise ValueError("Need at least 2 samples in each group to estimate sigma.")

    s2_D  = np.var(y_D, ddof=1)
    s2_Dp = np.var(y_Dp, ddof=1)

    nu = (m - 1) + (n - 1)  # degrees of freedom
    sp2 = ((m - 1) * s2_D + (n - 1) * s2_Dp) / nu

    # One-sided upper bound on sigma^2 via chi-square (use LOWER quantile!)
    chi2_lower = chi2.ppf(alpha, nu)
    if chi2_lower <= 0:
        return 0.0

    sigma2_up = nu * sp2 / chi2_lower
    sigma_up = np.sqrt(sigma2_up)

    mu_lo = sens / sigma_up
    return float(max(mu_lo, 0.0)), float(sigma_up)


def _conf_upper_binom_cp(k, n, alpha_one_sided=0.05):
    """
    Upper bound for a binomial proportion using Clopper–Pearson.

    alpha_one_sided is the desired one-sided error rate.
    We approximate this by taking the upper endpoint of a two-sided CI
    with confidence_level = 1 - alpha_one_sided (conservative but standard).
    """
    if n <= 0:
        return 1.0
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=1 - alpha_one_sided, method="exact")
    return ci.high


######
class JointBetaMu:
    """
    Inspired from Bayesian Estimation of Differential Privacy (https://arxiv.org/abs/2206.05199)
    and https://github.com/microsoft/responsible-ai-toolbox-privacy/blob/66d2d45b8f57683b0390cfa63774abb70235e5da/privacy_estimates/joint_density.py#L118

    Joint-beta (Jeffreys) model for (FPR, FNR) + μ-GDP region inversion.

    Posterior:
      FPR ~ Beta(0.5+FP, 0.5+TN)
      FNR ~ Beta(0.5+FN, 0.5+TP)
    independent.
    """
    def __init__(self, fp, tn, fn, tp):
        self.fpr_post = beta_dist(0.5 + fp, 0.5 + tn)
        self.fnr_post = beta_dist(0.5 + fn, 0.5 + tp)

    def prob_mu_private(self, mu, epsabs=1e-6):
        """
        Probability mass of μ-GDP feasible region:
          fnr >= beta_from_mu(fpr, mu)
        under independent posteriors.
        """
        def integrand(fpr):
            b = get_beta_from_gdp(fpr, mu)
            return self.fpr_post.pdf(fpr) * (1.0 - self.fnr_post.cdf(b))
        
        p, _ = integrate.quad(integrand, 0.0, 1.0, epsabs=epsabs)
        return float(np.clip(p, 0.0, 1.0))

    def mu_lo(self, alpha=0.1, xtol=1e-3, max_mu=50.0):
        """
        Returns μ_lo such that the μ-GDP region contains alpha posterior mass.
        This mirrors the epsilon_estimation.DensityModel().eps_lo convention.
        """
        assert 0 < alpha < 1

        def objective(mu):
            return self.prob_mu_private(mu, epsabs=max(xtol/5, 1e-6)) - alpha

        # If even μ=0 already contains >= alpha mass, lower bound is 0
        if objective(0.0) >= 0.0:
            return 0.0

        lo, hi = 0.0, 1.0
        while objective(hi) < 0.0:
            hi *= 2
            if hi >= max_mu:
                hi = max_mu
                break

        # If still not enough mass even at max_mu, return max_mu (very conservative)
        if objective(hi) < 0.0:
            return float(max_mu)

        res = root_scalar(objective, bracket=[lo, hi], xtol=xtol, method="brentq")
        return float(res.root)
######


def _threshold_grid_from_scores(
    valid_scores,
    mode="quantiles",  # ["all_unique", "quantiles"]
    n_thresholds=200,
):
    if mode == "all_unique":
        thresholds = np.unique(valid_scores)
    elif mode == "quantiles":
        qs = np.linspace(0, 1, n_thresholds)
        thresholds = np.unique(np.quantile(valid_scores, qs))
    else:
        raise ValueError("threshold_mode must be 'all_unique' or 'quantiles'")
    return thresholds


def _eval_thresholds(scores_out, scores_in, thresholds, ci_method="bonferroni_cp", alpha=0.1):
    """Compute confusion/rates for each threshold."""
    P = len(scores_in)
    N = len(scores_out)

    TP, FN, FP, TN = [], [], [], []
    FPR, FNR, ADV, MU_HAT, MU_LOWER = [], [], [], [], []

    for t in thresholds:
        tp, fn, fp, tn = _confusion_at_threshold(scores_out, scores_in, float(t))
        fpr, fnr, adv = _rates_from_confusion(tp, fn, fp, tn)
        mu_hat = _mu_from_fpr_fnr(fpr, fnr)
        mu_lower = _mu_lo_from_counts(tp=tp, fn=fn, fp=fp, tn=tn, ci_method=ci_method, alpha=alpha) 

        TP.append(tp)
        FN.append(fn)
        FP.append(fp)
        TN.append(tn)
        FPR.append(fpr)
        FNR.append(fnr)
        ADV.append(adv)
        MU_HAT.append(mu_hat)
        MU_LOWER.append(mu_lower)

    return {
        "thresholds": thresholds.astype(float),
        "TP": np.array(TP, dtype=int),
        "FN": np.array(FN, dtype=int),
        "FP": np.array(FP, dtype=int),
        "TN": np.array(TN, dtype=int),
        "FPR": np.array(FPR, dtype=float),
        "FNR": np.array(FNR, dtype=float),
        "advantage": np.array(ADV, dtype=float),
        "mu_hat": np.array(MU_HAT, dtype=float),
        "mu_lower": np.array(MU_LOWER, dtype=float),
        "P": np.array([P], dtype=int)[0],
        "N": np.array([N], dtype=int)[0],
    }


def _confusion_at_threshold(scores_out, scores_in, t):
    """Return (TP, FN, FP, TN) when predicting 'in' if score>=t."""
    P = len(scores_in)
    N = len(scores_out)
    tp = int(np.sum(scores_in >= t))
    fn = int(P - tp)
    fp = int(np.sum(scores_out >= t))
    tn = int(N - fp)
    return tp, fn, fp, tn


def _rates_from_confusion(tp, fn, fp, tn):
    """Return (FPR, FNR, advantage) where advantage = TPR - FPR."""
    P = tp + fn
    N = tn + fp
    fpr = fp / max(N, 1)
    fnr = fn / max(P, 1)
    tpr = tp / max(P, 1)
    adv = tpr - fpr
    return float(fpr), float(fnr), float(adv)


def _mu_from_fpr_fnr(fpr, fnr):
    """Compute μ from a single (FPR,FNR) point."""
    clip_eps = 1e-6
    fpr = np.clip(fpr, clip_eps, 1 - clip_eps)
    fnr = np.clip(fnr, clip_eps, 1 - clip_eps)
    mu = norm.ppf(1 - fpr) - norm.ppf(fnr)
    mu = np.clip(mu, 0, None)
    return mu


def _select_optimal_threshold(curve, threshold_selection):
    """Select optimal threshold from cureve (validation curve)."""
    if threshold_selection == "max_advantage":
        idx = int(np.argmax(curve["advantage"]))
    elif threshold_selection == "max_mu_hat":
        idx = int(np.argmax(curve["mu_hat"]))
    elif threshold_selection == "max_mu_lower":
        idx = int(np.argmax(curve["mu_lower"]))
    else:
        raise ValueError("threshold_selection must be 'max_advantage', 'max_mu_hat' or 'max_mu_lower'")

    t = float(curve["thresholds"][idx])
    return t


def _mu_lo_from_counts(tp, fn, fp, tn,
    ci_method="bonferroni_cp",    # "bonferroni_cp", "joint_beta"
    alpha=0.1,
):
    """Compute a single μ lower bound from one confusion tuple."""
    P = tp + fn
    N = tn + fp

    if ci_method == "bonferroni_cp":
        # Bonferroni across (FPR,FNR): alpha/2 per rate
        # fix fpr for highest advantage; get ci for fnr with all alpha
        alpha_each = alpha / 2
        fpr_u = _conf_upper_binom_cp(fp, N, alpha_one_sided=alpha_each)
        fnr_u = _conf_upper_binom_cp(fn, P, alpha_one_sided=alpha_each)
        return _mu_from_fpr_fnr(fpr_u, fnr_u)

    if ci_method == "joint_beta":
        jb = JointBetaMu(fp=fp, tn=tn, fn=fn, tp=tp)
        return jb.mu_lo(alpha=alpha)

    raise ValueError("ci_method must be 'bonferroni_cp' or 'joint_beta'")


def run_audit(
    out_data,
    in_data,
    n_train,
    n_valid,
    n_test,
    classifier="xgboost",                 # "xgboost", "random_forest"
    threshold_mode="quantiles",           # "all_unique", "quantiles"
    n_thresholds=200,
    threshold_selection="max_advantage",  # "max_advantage", "max_mu_hat", "max_mu_lower"
    ci_method="joint_beta",               # "bonferroni_cp", "joint_beta"
    alpha=0.1,
    random_state=None,
):
    """
    1) Train attack model on TRAIN split.
    2) Compute scores on VALID and TEST (but DO NOT use TEST for threshold selection).
    3) Choose threshold t* using VALID only by objective:
       - max_advantage: maximize TPR - FPR
       - max_mu_hat: maximize μ_hat computed from (FPR,FNR)
    4) Return artifacts + full VALID diagnostics curves for plotting.
    """

    # --- train attack model ---
    X_train = np.concatenate([out_data[:n_train], in_data[:n_train]])
    y_train = np.array([0] * n_train + [1] * n_train)

    if classifier == "xgboost":
        clf = GradientBoostingClassifier(random_state=random_state)
    elif classifier == "random_forest":
        clf = RandomForestClassifier(random_state=random_state)
    else:
        raise ValueError("classifier must be 'xgboost' or 'random_forest'")
    clf.fit(X_train, y_train)

    # --- compute scores for valid+test ---
    out_scores_all = clf.predict_proba(out_data[n_train:])[:, 1]
    in_scores_all = clf.predict_proba(in_data[n_train:])[:, 1]

    out_scores_valid = out_scores_all[:n_valid]
    out_scores_test = out_scores_all[n_valid:n_valid + n_test]
    in_scores_valid = in_scores_all[:n_valid]
    in_scores_test = in_scores_all[n_valid:n_valid + n_test]
    
    # --- AUC diagnostics (do NOT use for selection) ---
    # valid auc
    y_valid = np.array([0] * n_valid + [1] * n_valid)
    scores_valid = np.concatenate([out_scores_valid, in_scores_valid])
    auc_valid = roc_auc_score(y_valid, scores_valid)
    auc_valid = max(auc_valid, 1 - auc_valid)

    # test auc
    y_test = np.array([0] * n_test + [1] * n_test)
    scores_test = np.concatenate([out_scores_test, in_scores_test])
    auc_test = roc_auc_score(y_test, scores_test)
    auc_test = max(auc_test, 1 - auc_test)

    # valid/test auc
    y_vt = np.array([0] * (n_valid + n_test) + [1] * (n_valid + n_test))
    scores_vt = np.concatenate([out_scores_all, in_scores_all])
    auc_vt = roc_auc_score(y_vt, scores_vt)
    auc_vt = max(auc_vt, 1 - auc_vt)
    
    # --- extract thresholds grid from VALID only ---
    valid_scores = np.concatenate([out_scores_valid, in_scores_valid])
    thresholds = _threshold_grid_from_scores(valid_scores, mode=threshold_mode, n_thresholds=n_thresholds)

    # --- evaluate curve and select threshold on VALID only ---
    valid_curve = _eval_thresholds(out_scores_valid, in_scores_valid, thresholds, ci_method=ci_method, alpha=alpha)
    opt_t = _select_optimal_threshold(valid_curve, threshold_selection)
    valid_curve["opt_t"] = opt_t
    
    # --- evaluate/estimate on TEST ---
    tp, fn, fp, tn = _confusion_at_threshold(out_scores_test, in_scores_test, opt_t)
    fpr, fnr, adv = _rates_from_confusion(tp, fn, fp, tn)
    mu_hat = _mu_from_fpr_fnr(fpr, fnr)

    mu_lower = _mu_lo_from_counts(
        tp=tp, fn=fn, fp=fp, tn=tn,
        ci_method=ci_method,
        alpha=alpha,
    )
    
    return {
        "valid_test": {
            "auc": auc_vt,
        },
        "valid": {
            "auc": auc_valid,
            "curve": valid_curve,
        },
        "test": {
            "auc": auc_test,
            "point": {
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "FPR": fpr,
                "FNR": fnr,
                "advantage": adv,
                "mu_hat": mu_hat,
                "mu_lower": mu_lower,
            },
        },
    }


