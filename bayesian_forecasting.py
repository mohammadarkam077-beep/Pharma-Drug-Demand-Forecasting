"""
Bayesian Hierarchical Forecasting Model (OPTIMIZED)
====================================================
Uses PyMC + nutpie for fast probabilistic state-level forecasting
with proper uncertainty quantification.
"""

import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt

# Try to use nutpie (3-10x faster, no g++ needed)
try:
    import nutpie
    NUTS_SAMPLER = "nutpie"
    print("✓ Using nutpie sampler (fast Rust-based NUTS)")
except ImportError:
    NUTS_SAMPLER = "pymc"
    print("⚠ nutpie not installed — falling back to default PyMC sampler")
    print("  Install with: pip install nutpie")


class BayesianPharmaForecaster:
    """
    Hierarchical Bayesian model:
    - Level 1: National parameters (priors from literature)
    - Level 2: Regional effects (North, South, East, West)
    - Level 3: State-specific deviations
    """

    def __init__(self, state_data, n_samples=1000, n_tune=500):
        self.data = state_data.reset_index(drop=True).copy()
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.trace = None
        self.model = None

        # Pre-compute & standardize covariates (helps geometry massively)
        access = self.data['Healthcare_Access_Score'].values.astype(float)
        self.access_std = (access - access.mean()) / access.std()

        # Region encoding
        self.region_codes, self.region_labels = pd.factorize(self.data['Region'])
        self.n_regions = len(self.region_labels)
        self.n_states = len(self.data)

    def build_model(self):
        """Define the Bayesian model with non-centered parameterization"""
        with pm.Model() as model:
            # === HYPERPRIORS (national level) ===
            # National mean prevalence on logit scale (much better geometry than Beta)
            mu_prev_logit = pm.Normal('mu_prev_logit', mu=-2.4, sigma=0.5)  # ~8% mean
            sigma_region = pm.HalfNormal('sigma_region', sigma=0.3)
            sigma_state = pm.HalfNormal('sigma_state', sigma=0.3)

            # === REGIONAL EFFECTS (non-centered parameterization) ===
            region_offset_raw = pm.Normal(
                'region_offset_raw', mu=0, sigma=1, shape=self.n_regions
            )
            region_offset = pm.Deterministic(
                'region_offset', region_offset_raw * sigma_region
            )

            # === STATE-LEVEL EFFECTS (non-centered) ===
            state_offset_raw = pm.Normal(
                'state_offset_raw', mu=0, sigma=1, shape=self.n_states
            )
            state_offset = pm.Deterministic(
                'state_offset', state_offset_raw * sigma_state
            )

            # State prevalence on logit scale → transform to probability
            state_logit = (
                mu_prev_logit
                + region_offset[self.region_codes]
                + state_offset
            )
            state_prev = pm.Deterministic('state_prev', pm.math.sigmoid(state_logit))

            # === DIAGNOSIS RATE (logit-linear in access score) ===
            access_intercept = pm.Normal('access_intercept', mu=0.0, sigma=1.0)
            access_effect = pm.Normal('access_effect', mu=0.3, sigma=0.2)

            diagnosis_logit = access_intercept + access_effect * self.access_std
            diagnosis_rate = pm.Deterministic(
                'diagnosis_rate', pm.math.sigmoid(diagnosis_logit)
            )

            # === LIKELIHOOD ===
            adult_pop = self.data['Adult_Pop'].values.astype(float)
            observed = self.data['Diagnosed_Patients'].values.astype(float)

            expected = adult_pop * state_prev * diagnosis_rate

            # Use a more reasonable noise model — observation noise scales with sqrt
            # (Poisson-like) but bounded; or use lognormal for strictly positive counts
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=observed.std())

            pm.Normal(
                'patients_obs',
                mu=expected,
                sigma=sigma_obs,
                observed=observed
            )

            self.model = model

        print(f"✓ Bayesian model built")
        print(f"  States: {self.n_states} | Regions: {self.n_regions}")
        print(f"  Using non-centered parameterization for stability")
        return self

    def sample(self):
        """MCMC sampling using nutpie (fast) or PyMC fallback"""
        print(f"\n🔬 Running MCMC sampling ({self.n_samples} draws, {self.n_tune} tune)...")
        print(f"   Sampler: {NUTS_SAMPLER}")

        with self.model:
            self.trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                chains=4,
                cores=1,                     # avoid Windows multiprocessing issues
                target_accept=0.95,          # reduces divergences
                nuts_sampler=NUTS_SAMPLER,
                return_inferencedata=True,
                progressbar=True,
                random_seed=42,
            )
        print(f"✓ Sampling complete")
        return self

    def diagnose(self):
        """Check model convergence"""
        print(f"\n📊 MODEL DIAGNOSTICS:")
        summary = az.summary(
            self.trace,
            var_names=['mu_prev_logit', 'sigma_region', 'sigma_state',
                       'region_offset', 'access_effect']
        )
        print(summary)

        max_rhat = summary['r_hat'].max()
        if max_rhat < 1.05:
            print(f"\n✅ Model converged (max R-hat = {max_rhat:.3f})")
        else:
            print(f"\n⚠ Convergence issue (max R-hat = {max_rhat:.3f})")

        # Divergence count
        try:
            n_div = int(self.trace.sample_stats['diverging'].sum())
            total = self.trace.sample_stats['diverging'].size
            print(f"   Divergences: {n_div} / {total} ({100*n_div/total:.2f}%)")
        except Exception:
            pass

        return self

    def plot_posteriors(self, output_path='outputs/bayesian_posteriors.png'):
        """Plot posterior distributions"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        az.plot_posterior(
            self.trace,
            var_names=['mu_prev_logit', 'sigma_region', 'sigma_state', 'access_effect'],
            hdi_prob=0.95
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
        return self

    def get_state_credible_intervals(self):
        """Get 95% credible intervals for each state's prevalence"""
        state_prev_samples = self.trace.posterior['state_prev'].values
        # Shape: (chains, draws, n_states)

        intervals = pd.DataFrame({
            'State': self.data['State'].values,
            'Region': self.data['Region'].values,
            'Mean_Prevalence': state_prev_samples.mean(axis=(0, 1)),
            'Lower_95_CI':   np.percentile(state_prev_samples, 2.5,  axis=(0, 1)),
            'Upper_95_CI':   np.percentile(state_prev_samples, 97.5, axis=(0, 1)),
            'Std':           state_prev_samples.std(axis=(0, 1))
        })

        intervals['Uncertainty'] = intervals['Upper_95_CI'] - intervals['Lower_95_CI']
        return intervals.sort_values('Mean_Prevalence', ascending=False)


# === USAGE ===
if __name__ == "__main__":
    state_data = pd.read_csv(r"D:\Forecasting\data\state_master.csv")

    bayesian = (BayesianPharmaForecaster(state_data, n_samples=1000, n_tune=500)
                .build_model()
                .sample()
                .diagnose()
                .plot_posteriors(r"D:\Forecasting\outputs\bayesian_posteriors.png"))

    intervals = bayesian.get_state_credible_intervals()
    intervals.to_csv(r"D:\Forecasting\outputs\bayesian_state_intervals.csv", index=False)

    print(f"\n📊 Top 5 states with HIGHEST uncertainty:")
    print(intervals.nlargest(5, 'Uncertainty').to_string(index=False))

    print(f"\n📊 Top 5 states with HIGHEST prevalence:")
    print(intervals.head(5).to_string(index=False))