data {
  int<lower=1> N;                 // number of athletes
  int<lower=1> J;                 // total number of unique ages
  int<lower=1> max_obs;           // maximum number of observed components across all athletes
  int<lower=1> obs_count[N];  
  int<lower=1, upper=J> age_idx[N, max_obs];
  real y[N, max_obs];     // matrix of observations for each subject (only first obs_count[n] are valid)
}

parameters {
  corr_matrix[J] Omega; 
  vector<lower=0>[J] sigma;
  matrix[N, J] gamma; 
  real<lower=0> sigma_y; // Residual (observation) noise standard deviation
  vector[J] gamma_0; // population mean effects
}

transformed parameters {
  cov_matrix[J] Sigma; 
  Sigma = quad_form_diag(Omega, sigma);
}

model{
  sigma ~ cauchy(0, 5); // prior on the standard deviations
  Omega ~ lkj_corr(1); // LKJ prior on the correlation matrix
  gamma_0 ~ normal(0, 1000);
  // For each athlete, impose MN prior only on the observed ages
  for (i in 1:N) {
    int k = obs_count[i];  // number of observed components for athlete i
    // use age indices to get random effect
    vector[k] gamma_i = gamma[i, age_idx[i, 1:k]]';  // transpose to get a column vector
    gamma_i ~ multi_normal(gamma_0[age_idx[i, 1:k]], Sigma[age_idx[i, 1:k], age_idx[i, 1:k]]); // so we get a submatrix of covariances between observed ages of particular athlete
  }
  // prior for residual noise
  sigma_y ~ cauchy(0, 5);
  // Likelihood for the observed y values.
  for (i in 1:N) {
    for (j in 1:obs_count[i]) {
      // Get the index into the full random-effect vector
      int a = age_idx[i, j];
      // The observed outcome is modeled as the sum of the random effect and noise.
      y[i, j] ~ normal(gamma[i, a], sigma_y);
    }
  }
}
