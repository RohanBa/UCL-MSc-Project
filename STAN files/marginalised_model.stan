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
  sigma_y ~ cauchy(0, 5); // prior for residual noise

  for (n in 1:N) {
    int k = obs_count[n];
    vector[k] mu_n;
    matrix[k, k] Sigma_n;

    for (i in 1:k) {
      mu_n[i] = gamma_0[age_idx[n, i]];
    }

    for (i in 1:k) {
      for (j in 1:k) {
        Sigma_n[i, j] = Sigma[age_idx[n, i], age_idx[n, j]];
      }
    }

    to_vector(y[n, 1:k]) ~ multi_normal(mu_n, Sigma_n + diag_matrix(rep_vector(square(sigma_y), k)));
  }
}

