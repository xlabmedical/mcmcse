# MCMCSE

Provides tools for computing Monte Carlo standard errors (MCSE) in Markov chain Monte Carlo (MCMC). A python/numpy implementation of [mcmcse.r](https://cran.r-project.org/web/packages/mcmcse/index.html), extended to support numpy broadcasting.

It implements the following methods:

  * [mcse](#mcse):
    * Computes the MCMC estimate of expectation of g, with standard error.
  * [mcse_mat](#mcse_mat):
    * Only for mimicking mcmcse.R, as the function mcse supports numpy arrays of any shape. See [mcse](#mcse).
  * [mcse_p](#mcse_p):
    * Computes the MCMC estimate of percentile p, with standard error.
  * [mcse\_p\_mat](#mcse\_p\_mat):
    * Only for mimicking mcmcse.r, as the function mcse_p supports numpy arrays of all shapes. See [mcse_p](#mcse_p).
  * [mcse_q](mcse_q):
    * Computes the MCMC estimate of quantile q, with standard error.
  * [mcse\_q\_mat](mcse\_q\_mat):
    * Only for mimicking mcmcse.r, as the function mcse_q supports numpy arrays of all shapes. See [mcse_q](mcse_q).
  * [ess](ess):
    * Estimate effective sample size (ESS) as described in Gong and Felgal (2015).


See bellow for documentation.

## Methods

### mcse

  Computes the MCMC estimate of expectation of g, with standard error.
  NOTE: function broadcasts over numpy arrays
  
  **Arguments**:
  * x: data
      * if one sample is of shape (n1,n2, ..., nk)
        than x.shape == (n_samples, n1,n2,...nk)
  * size: batch size
      * size in ["sqroot", "cuberoot"] or size is numeric
        default is sqroot
  * g: function which expectation is to be computed and estimated
      * if a sample is of shape (n1,...,nk), g has to handle such inputs
  * method: which method to use
      * method in ["bm", "obm", "tukey", "bartlett"]
        default is bm
  
  **Returns**:
  * est: estimated expectation of g
      * est.shape == (n1,n2,...nk)
  * ess: estimated standard error, of the expectation of g
      * ess.shape ==(n1,n2,...,nk)
  Note when they have dimension zero, numerics are returned

### mcse_mat

  Only for mimicking mcmcse.R, as the function mcse supports numpy arrays of any shape. It thus only forwards its arguments to mcse.
  See mcse.

### mcse_p

  Computes the MCMC estimate of percentile p, with standard error.
  NOTE: function broadcasts over numpy arrays

  **Arguments**:
  * x: data
      * if one sample is of shape (n1,n2, ..., nk)
        than x.shape == (n_samples, n1,n2,...nk)
  * p: percentile to compute
      * 0<=p<=100
  * size: batch size
      * size in ["sqroot", "cuberoot"] or size is numeric
        default is sqroot
  * g: function which percentiles are to be computed and estimated
      * if a sample is of shape (n1,...,nk), g has to handle such inputs
  * method: which method to use
      * method in ["bm", "obm", "sub"]
        default is bm

  **Returns**:
  * est: estimated percentile
      * est.shape == (n1,n2,...nk)
  * ess: estimated standard error of the percentile
      * ess.shape ==(n1,n2,...,nk)
  Note when they have dimension zero, numerics are returned

### mcse\_p\_mat

Only for mimicking mcmcse.r, as the function mcse\_p supports numpy arrays of all shapes. It thus only forwards its arguments to mcse\_p.
See mcse_p

### mcse_q

  Computes the MCMC estimate of quantile q, with standard error.
  NOTE: function broadcasts over numpy arrays
 
  **Arguments**:
  * x: data
      * if one sample is of shape (n1,n2, ..., nk)
        than x.shape == (n_samples, n1,n2,...nk)
  * q: quantile to compute
      * 0<=q<=100
  * size: batch size
      * size in ["sqroot", "cuberoot"] or size is numeric
        default is sqroot
  * g: function which quantiles are to be computed and estimated
      * if a sample is of shape (n1,...,nk), g has to handle such inputs
  * method: which method to use
      * method in ["bm", "obm", "sub"]
        default is bm

  **Returns**:
  * est: estimated quantile
      * est.shape == (n1,n2,...nk)
  * ess: estimated standard error of the quantile
      * ess.shape ==(n1,n2,...,nk)
  Note when they have dimension zero, numerics are returned

### mcse\_q\_mat

Only for mimicking mcmcse.r, as the function mcse\_q supports numpy arrays of all shapes. It thus only forwards its arguments to mcse\_q.
See mcse_q.

### ess

  Estimate effective sample size (ESS) as described in Gong and Felgal (2015).
  
  **Arguments**:
  * x: data
      * if one sample is of shape (n1,n2, ..., nk)
        than x.shape == (n_samples, n1,n2,...nk)
  * g: function which expectation is to be computed and estimated
      * if a sample is of shape (n1,...,nk), g has to handle such inputs
  * kwargs: arguments to be passed to mcse

  **Returns**:
  * ess: estimated sample size