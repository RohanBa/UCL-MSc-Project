data {
  int<lower=1> N;         // number of athletes (here 200)
  int<lower=1> J;         // number of unique ages (here 5)
  // Observed outcomes: each row is an athlete
  real y[N, J];     
  // The sorted vector of unique ages (22,23,24,25,26).
  vector[J] unique_ages;  
}
parameters {
  // LKJ prior for the full J x J correlation matrix.
  corr_matrix[J] Omega; 
  // Standard deviations for the random effects.
  vector<lower=0>[J] sigma; 
  // Subject-specific random effects (each athlete has a J-dimensional vector).
  matrix[N, J] gamma;  
  // Residual error standard deviation.
  real<lower=0> sigma_y;
  // population mean effects
  vector[J] gamma_0;            
}
transformed parameters {
  // Build the full covariance matrix for the random effects.
  cov_matrix[J] Sigma;
  Sigma = quad_form_diag(Omega, sigma);
}
model {
  // Priors on covariance parameters and residual noise.
  sigma ~ cauchy(0, 5);
  Omega ~ lkj_corr(1);
  sigma_y ~ cauchy(0, 5);
  gamma_0 ~ normal(0, 10000); //rep_vector(0, J)
  
  // Prior for the subject-specific random effects.
  for (n in 1:N) {
    gamma[n] ~ multi_normal(gamma_0, Sigma);
  }
  
  // Likelihood for the observed outcomes.
  for (n in 1:N) {
   for (j in 1:J) {
      y[n, j] ~ normal(gamma[n, j], sigma_y);
    }
  }
}
