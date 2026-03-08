data {
  int<lower=1> N;                 // number of athletes
  int<lower=1> J;                 // total number of unique ages
  int<lower=1> max_obs;           // maximum number of observed components across all athletes
  int<lower=1> obs_count[N];  
  int<lower=1, upper=J> age_idx[N, max_obs];
  vector[J] unique_ages;          // sorted unique ages
  real center_age;                // centering
  real y[N, max_obs];             // observed performances
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
  vector[5] beta;                  // fixed effects for age polynomial
  matrix[N, J] gamma;              // athlete-specific deviations
  real<lower=0> sigma_1;            // SD for initial gamma
  real<lower=0> sigma_2;            // SD for initial gamma
  real<lower=0> tau;               // SD for random walk increments
  real<lower=2> nu0;              // dof for initial state  (>=2 keeps variance finite)
  real<lower=2> nu;               // dof for increments     (>=2 keeps variance finite)
  real<lower=0> sigma_y;           // residual SD
}

model {
  // Priors
  beta ~ normal(0, 100);
  sigma_1 ~ cauchy(0, 2.5);
  sigma_2 ~ cauchy(0, 2.5);
  tau ~ cauchy(0, 2.5);
  nu0 ~ exponential(0.2);  // favors small dof (heavy tails) but allows larger
  nu ~ exponential(0.2);
  sigma_y ~ cauchy(0, 2.5);
  
  // Random walk prior for gamma
  for (n in 1:N) {
    gamma[n, 1] ~ student_t(nu0, 0, sigma);  // initial deviation
    for (j in 2:J) {
      gamma[n, j] ~ student_t(nu, gamma[n,j-1], tau);  // RW increments
    }
  }
  
  // Likelihood over observed entries
  for (n in 1:N) {
    int k = obs_count[n];
    vector[k] mu;
    vector[k] y_obs_n;
    
    for (i in 1:k) {
      int a_idx = age_idx[n, i];
      mu[i] = dot_product(row(X_poly, a_idx), beta) + gamma[n, a_idx];
      y_obs_n[i] = y[n, i];
    }
    
    y_obs_n ~ normal(mu, sigma_y);
  }
}


