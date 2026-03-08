data {
  int<lower=1> N;         // number of athletes 
  int<lower=1> J;         // number of unique ages 
  // Observed outcomes: each row is an athlete
  real y[N, J];     
  // The sorted vector of unique ages 
  vector[J] unique_ages;  
  int<lower=0, upper=1> missing_positions[N, J]; // 1 if missing, 0 if observed
  real center_age;     // age to center the polynomial, e.g 27
}

transformed data {
  vector[J] centered_ages = unique_ages - center_age;  // Now centered at `center_age`
  matrix[J, 5] X_poly;                                // Polynomial terms (no intercept)
  for (j in 1:J) {
    X_poly[j, 1] = 1;                                 // Intercept column
    X_poly[j, 2] = centered_ages[j];                  // Linear term
    X_poly[j, 3] = centered_ages[j]^2;                // Quadratic term
    X_poly[j, 4] = centered_ages[j]^3;                // Cubic term
    X_poly[j, 5] = centered_ages[j]^4;                // Quartic term
  }
}

parameters {
  // LKJ prior for the full J x J correlation matrix.
  corr_matrix[J] Omega; 
  // Standard deviations for the random effects, prior scale
  vector<lower=0>[J] sigma; 
  // Subject-specific random effects (each athlete has a J-dimensional vector).
  matrix[N, J] gamma;  
  // Global intercept
  real beta0;
  // Polynomial coefficients 
  vector[4] beta;
  // Error scale/standard deviation
  real<lower=0> sigma_y;
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
  beta0 ~ normal(0, 10^3);
  beta ~ normal(0, 2);
  
   // Prior for the subject-specific random effects.
  for (n in 1:N) {
    gamma[n] ~ multi_normal(rep_vector(0, J), Sigma);
  }
  
  // Likelihood for the observed outcomes.
  for (n in 1:N) {
    for (j in 1:J) {
      if (!missing_positions[n, j]) {  // Only model if not missing
        real mu = beta0 + X_poly[j,2:5] * beta + gamma[n,j];
        y[n, j] ~ normal(mu, sigma_y);
      }
    }
  }
}
