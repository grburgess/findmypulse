

real partial_log_like(int[] counts_slice, int start, int end, vector time, vector log_exposure,
		      row_vector omega1_src,
		      row_vector omega2_src,
		      vector beta1_src,
		      vector beta2_src,
		      vector scale_src,


		      row_vector omega1_bkg,
		      row_vector omega2_bkg,
		      vector beta1_bkg,
		      vector beta2_bkg,
		      vector scale_bkg,


		      row_vector omega1_p,
		      row_vector omega2_p,
		      vector beta1_p,
		      vector beta2_p,
		      vector scale_p,


		      ) {

  int N = size(counts_slice);
  real lp = 0;
  vector[N] log_exposure_slice;
  vector[N] log_src_rate;
  vector[N] log_bkg_rate;
  vector[N] p;

  log_src_rate = ((scale_src[1] * cos(time[start:end] * omega1_src) + scale_src[2] * cos(time[start:end] * omega1_src)  ) * beta1_src) + ((scale_src[2] * sin(time[start:end] * omega1_src) + scale_src[2] * sin(time[start:end] * omega2_src)  ) * beta2_src);

  log_bkg_rate = ((scale_bkg[1] * cos(time[start:end] * omega1_bkg) + scale_bkg[2] * cos(time[start:end] * omega1_bkg)  ) * beta1_bkg) + ((scale_bkg[2] * sin(time[start:end] * omega1_bkg) + scale_bkg[2] * sin(time[start:end] * omega2_bkg)  ) * beta2_bkg);
  
  p = inv_logit(-3 + ((scale_p[1] * cos(time[start:end] * omega1_p) + scale_p[2] * cos(time[start:end] * omega1_p)  ) * beta1_p) + ((scale_p[2] * sin(time[start:end] * omega1_p) + scale_p[2] * sin(time[start:end] * omega2_p)  ) * beta2_p));

  for (n in 1:N) {
    lp += log_mix(p[n],
		  poisson_log_propto_lpmf(counts_slice[n] | log_exposure_slice[n] + log_sum_exp(log_src_rate[n], log_bkg_rate[n])),
		  log_exposure_slice[n] + log_bkg_rate[n]		  
		  );
  }
  

  return lp;

}


