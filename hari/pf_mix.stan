functions {
#include "functions.stan"
}

data {
  int<lower=1> N; // number of time bins for LC 1
  vector[N] time; // mid-points of LC 1
  int counts[N]; // counts in LC 1

  vector[N] exposure;

  //  real bw_src;

  int<lower=1> K_src; // number of FFs
  int<lower=1> K_bkg; // number of FFs
  int<lower=1> K_p; //

  int grainsize;
}
transformed data {



  vector[N] log_exposure = log(exposure);

  real max_range = max(time) - min(time);
  

}
parameters {


  row_vector[K_src] omega_src_var[2]; // this weird MC integration thing. I suppose I could do this in stan
  row_vector[K_bkg] omega_bkg_var[2]; // this weird MC integration thing. I suppose I could do this in stan
  row_vector[K_p] omega_p_var[2]; //

  
  vector[K_src] beta1_src; // the amplitude along the cos basis
  vector[K_bkg] beta1_bkg; // the amplitude along the cos basis
  vector[K_p] beta1_p; // the amplitude along the cos basis
  vector[K_src] beta2_src; // the amplitude along the cos basis
  vector[K_bkg] beta2_bkg; // the amplitude along the cos basis
  vector[K_p] beta2_p; // the amplitude along the cos basis



  vector[2] log_scale_src;
  vector[2] log_scale_bkg;
  vector[2] log_scale_p;



  
  real<lower=.5, upper=1> range1_bkg_raw;
  real<lower=.5, upper=range1_bkg_raw> range2_bkg_raw;

  real<lower=0, upper=.5> range1_src_raw;
  real<lower=0, upper=range1_src_raw> range2_src_raw;
  
  real<lower=0, upper=.8> range1_p_raw;
  real<lower=0, upper=range1_p_raw> range2_p_raw;

}


transformed parameters {

  vector[2] range_src;
  vector[2] range_bkg;
  vector[2] range_p;
  
  vector[2] bw_src;
  vector[2] bw_bkg;
  vector[2] bw_p;

  row_vector[K_src] omega_src[2];
  row_vector[K_bkg] omega_bkg[2];
  row_vector[K_p] omega_p[2];
  
  vector[2] scale_src = exp(log_scale_src) * inv_sqrt(K_src);
  vector[2] scale_bkg = exp(log_scale_bkg) * inv_sqrt(K_bkg);
  vector[2] scale_p = exp(log_scale_p) * inv_sqrt(K_p);


  range_src[1] = range1_src_raw * max_range;
  range_src[2] = range2_src_raw * max_range;

  range_bkg[1] = range1_bkg_raw * max_range;
  range_bkg[2] = range2_bkg_raw * max_range;

  range_p[1] = range1_p_raw * max_range;
  range_p[2] = range2_p_raw * max_range;

  bw_src = inv(range_src);
  bw_bkg = inv(range_bkg);
  bw_p = inv(range_p);

  
  omega_src[1] = omega_src_var[1] * bw_src[1];
  omega_src[2] = omega_src_var[2] * bw_src[2];

  omega_bkg[1] = omega_bkg_var[1] * bw_bkg[1];
  omega_bkg[2] = omega_bkg_var[2] * bw_bkg[2];

  omega_p[1] = omega_p_var[1] * bw_p[1];
  omega_p[2] = omega_p_var[2] * bw_p[2];
  

}

model {

  // priors

  beta1_src ~ std_normal();
  beta2_src ~ std_normal();

  beta1_bkg ~ std_normal();
  beta2_bkg ~ std_normal();

  beta1_p ~ std_normal();
  beta2_p ~ std_normal();

  // log scale
  
  log_scale_src[2] ~ normal(-2,.5);
  log_scale_src[1] ~ normal(-1,.5);

  log_scale_p[2] ~ normal(0,1);
  log_scale_p[1] ~ normal(0,1);
  
  log_scale_bkg ~ std_normal();


  // range
  
  range1_src_raw ~ normal(0, .25);
  range2_src_raw ~ normal(0, .25);

  range1_bkg_raw ~ normal(1, .25);
  range2_bkg_raw ~ normal(1, .25);

  range1_p_raw ~ normal(.5, .5);
  range2_p_raw ~ normal(.5, .5);


  // omega
  
  omega_src_var[1] ~ std_normal();
  omega_src_var[2] ~ std_normal();


  omega_bkg_var[1] ~ std_normal();
  omega_bkg_var[2] ~ std_normal();


  omega_p_var[1] ~ std_normal();
  omega_p_var[2] ~ std_normal();

  
  target += reduce_sum(partial_log_like, counts, grainsize,
                       time, log_exposure,
                       omega_src[1], omega_src[2], beta1_src, beta2_src,
                       scale_src, 
                       omega_bkg[1], omega_bkg[2], beta1_bkg, beta2_bkg,
                       scale_bkg, 
                       omega_p[1], omega_p[2], beta1_p, beta2_p, 
                       scale_p
                       );




}

generated quantities {

}
