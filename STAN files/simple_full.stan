data {
  int<lower=1> N;                 // number of athletes
  int<lower=1> J;                 // total number of unique ages
  int<lower=1> max_obs;           // maximum number of observed components across all athletes
  int<lower=1> obs_count[N];  
  int<lower=1, upper=J> age_idx[N, max_obs];
  vector[J] unique_ages;          // The sorted vector of unique ages 
  real center_age;                // age to center the polynomial, e.g 23
  real y[N, max_obs];             // matrix of observations for each subject (only first obs_count[n] are valid)
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
  vector[5] beta;
  vector<lower=0>[J] sigma; 
  corr_matrix[J] Omega; 
}

transformed parameters {
  // Build the full covariance matrix for the random effects.
  cov_matrix[J] Sigma;
  Sigma = quad_form_diag(Omega, sigma);
}

model {
  sigma ~ cauchy(0, 2.5);
  Omega ~ lkj_corr(1);
  beta ~ normal(0, 100);

  for (n in 1:N) {
    int k = obs_count[n];
    vector[k] mu;
    matrix[k, k] Sigma_n;

    for (i in 1:k) {
      mu[i] = dot_product(row(X_poly, age_idx[n, i]), beta);
      for (j in 1:k) {
        Sigma_n[i, j] = Sigma[age_idx[n, i], age_idx[n, j]];
      }
    }
    vector[k] y_obs_n;
    for (i in 1:k) {
      y_obs_n[i] = y[n, i];
    }
    y_obs_n ~ multi_normal(mu, Sigma_n);
  }
}
