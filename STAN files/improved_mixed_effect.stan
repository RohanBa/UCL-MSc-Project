data {
  int<lower=1> N;                  // number of athletes
  int<lower=1> J;                  // number of unique ages
  int<lower=1> max_obs;            // max observed components per athlete
  int<lower=1> obs_count[N];       // number of observed results per athlete
  int<lower=1, upper=J> age_idx[N, max_obs]; // age index for each obs
  vector[J] unique_ages;           // the sorted vector of unique ages
  real center_age;                 // age to center the polynomial
  real y[N, max_obs];              // observed performances
}

transformed data {
  vector[J] centered_ages = unique_ages - center_age;
  matrix[J, 5] X_poly;
  for (j in 1:J) {
    X_poly[j, 1] = 1;
    X_poly[j, 2] = centered_ages[j];
    X_poly[j, 3] = square(centered_ages[j]);
    X_poly[j, 4] = centered_ages[j]^3;
    X_poly[j, 5] = centered_ages[j]^4;
  }
}

parameters {
  vector[5] beta;                  // population polynomial coefficients
  vector[J] z_gamma[N];            // noncentered random effects (athlete-by-age)
  vector<lower=0>[J] sigma_gamma;  // stddevs for random effects
  corr_matrix[J] Omega;            // correlation for random effects
  real<lower=0> sigma_y;           // residual stddev
}

transformed parameters {
  cov_matrix[J] Sigma;
  matrix[J, J] L_Sigma;
  vector[J] gamma[N];

  Sigma = quad_form_diag(Omega, sigma_gamma);
  L_Sigma = cholesky_decompose(Sigma);

  for (n in 1:N)
    gamma[n] = L_Sigma * z_gamma[n];
}

model {
  // Priors
  beta ~ normal(0, 5); // reasonable prior for standardized predictors
  sigma_gamma ~ normal(0, 1); // random effects SD
  Omega ~ lkj_corr(0.5);        // slightly favors identity
  sigma_y ~ normal(0, 0.5);   // residual stddev

  // Noncentered parameterization
  for (n in 1:N)
    z_gamma[n] ~ normal(0, 1);

  // Likelihood (over observed entries only)
  for (n in 1:N) {
    int k = obs_count[n];
    vector[k] mu;
    vector[k] y_obs_n;

    for (i in 1:k) {
      int a_idx = age_idx[n, i];
      mu[i] = dot_product(row(X_poly, a_idx), beta) + gamma[n][a_idx];
      y_obs_n[i] = y[n, i];
    }
    y_obs_n ~ normal(mu, sigma_y);
  }
}
