import numpy as np
from scipy.stats import gaussian_kde

def mcse(x, size = "sqroot", g=None, method="bm"):
  """
  Computes the MCMC estimate of expectation of g, with standard error
  NOTE: function broadcasts over numpy arrays
  Args:
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
  Returns:
    * est: estimated expectation of g
        * est.shape == (n1,n2,...nk)
    * ess: estimated standard error, of the expectation of g
        * ess.shape ==(n1,n2,...,nk)
    Note when they have dimension zero, numerics are returned    
  """
  valid_methods=("bm", "obm", "tukey", "bartlett")
  valid_sizes=("sqroot","cuberoot")
  assert method in valid_methods, "%s in not a valid method"%method
  if not callable(g):
    g=lambda x:x
  n=len(x)
  shape=x.shape
  if size == "sqroot":
    b=np.floor(np.sqrt(n))
    a=np.floor(n/b)
  elif size == "cuberoot":
    b=np.floor(n**(1/3))
    a=np.floor(n/b)
  else:
    try:
      b=np.floor(size)
      a=np.floor(n/b)
    except TypeError:
      raise TypeError("Parameter size must be numeric if it is not 'sqroot' or 'cuberoot'")
  a=int(a)
  b=int(b)
  n_=a*b
  if method == "bm":
    g_x=g(x)
    y=np.mean(np.reshape(g_x[:n_],(a,b,*shape[1:])),axis=1)
    mu_hat=np.mean(g_x,axis=0)
    var_hat=b*np.sum((y-mu_hat)**2,axis=0)/(a-1)
    se=np.sqrt(var_hat/n)
  elif method == "obm":
    a=n-b
    y=np.stack([np.mean(g(x[k:k+b]),axis=0) for k in range(a)],axis=0)
    mu_hat=np.mean(g(x),axis=0)
    var_hat=n*b*np.sum((y-mu_hat)**2,axis=0)/(a-1)/a
    se=np.sqrt(var_hat/n)
  elif method == "tukey":
    alpha=np.arange(1,b+1)
    alpha=(1+np.cos(np.pi*alpha/b))/2*(1-alpha/n)
    mu_hat=np.mean(g(x),axis=0)
    print(1)
    R=np.stack([np.mean((g(x[:(n - j)]) - mu_hat) * (g(x[(j):n]) - mu_hat),axis=0) for j in range(b+1)],axis=0)
    perm=np.arange(1,len(R.shape))
    R_0=R[0]
    R_=np.transpose(R[1:],axes=(*perm,0))
    var_hat=R_0+2*np.sum(alpha*R_,axis=-1)
    se=np.sqrt(var_hat/n)
  else: #method == "bartlett"
    alpha=np.arange(1,b+1)
    alpha=(1-np.abs(alpha)/b)*(1-alpha/n)
    mu_hat=np.mean(g(x),axis=0)
    R=np.stack([np.mean((g(x[:(n - j)]) - mu_hat) * (g(x[(j):n]) - mu_hat),axis=0) for j in range(b+1)],axis=0)
    perm=np.arange(1,len(R.shape))
    R_0=R[0]
    R_=np.transpose(R[1:],axes=(*perm,0))
    var_hat=R_0+2*np.sum(alpha*R_,axis=-1)
    se=np.sqrt(var_hat/n)
  return mu_hat,se

def mcse_mat(*args,**kwargs):
  """
  Only for mimicking MCMCSE.R, as the function mcse supports all numpy arrays.
  It thus only forwards its arguments to mcse.
  See mcse
  """
  return mcse(*args,**kwargs)

def mcse_p(x, p, size = "sqroot", g=None, method="bm"):
  """
  Computes the MCMC estimate of percentile p, with standard error
  NOTE: function broadcasts over numpy arrays
  Args:
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
  Returns:
    * est: estimated percentile
        * est.shape == (n1,n2,...nk)
    * ess: estimated standard error of the percentile
        * ess.shape ==(n1,n2,...,nk)
    Note when they have dimension zero, numerics are returned    
  """
  assert 0<p<=100, "Percentile must be between 0 and 100"
  valid_methods=("bm", "obm", "sub")
  valid_sizes=("sqroot","cuberoot")
  assert method in valid_methods, "%s in not a valid method"%method
  if not callable(g):
    g=lambda x:x
  n=len(x)
  shape=x.shape
  if size == "sqroot":
    b=np.floor(np.sqrt(n))
    a=np.floor(n/b)
  elif size == "cuberoot":
    b=np.floor(n**(1/3))
    a=np.floor(n/b)
  else:
    try:
      b=np.floor(size)
      a=np.floor(n/b)
    except TypeError:
      raise TypeError("Parameter size must be numeric if it is not 'sqroot' or 'cuberoot'")
  a=int(a)
  b=int(b)
  n_=a*b
#
  counting=lambda X,x, axis=0: np.sum(X<=x,axis=axis)
  quant=lambda X, axis=0: np.percentile(X,p,axis=axis)
  if method == "bm":
    g_x=g(x)
    xi_hat=quant(g_x)
    y=np.mean(np.reshape(g_x[:n_]<=xi_hat,(a,b,*shape[1:])),axis=1)
    mu_hat=np.mean(y,axis=0)
    var_hat=b*np.sum((y-mu_hat)**2,axis=0)/(a-1)
    x_tmp=x.reshape((x.shape[0],-1)).T
    try:
      xi_hat_=xi_hat
      xi_hat[0]
    except IndexError:
      xi_hat_=np.array([xi_hat])
    f_hat=np.squeeze(np.stack(((gaussian_kde(sample))(hat) for sample,hat in zip(x_tmp,xi_hat_)),axis=0))
    f_hat=f_hat.reshape(x.shape[1:])
    se=np.sqrt(var_hat/n)/f_hat
  elif method == "obm":
    a=n-b
    xi_hat=quant(g(x))
    y=np.stack([np.mean(g(x[k:k+b])<=xi_hat,axis=0) for k in range(a)],axis=0)
    mu_hat=np.mean(y,axis=0)
    var_hat=n*b*np.sum((y-mu_hat)**2,axis=0)/(a-1)/a
    x_tmp=x.reshape((x.shape[0],-1)).T
    try:
      xi_hat_=xi_hat
      xi_hat[0]
    except IndexError:
      xi_hat_=np.array([xi_hat])
    f_hat=np.squeeze(np.stack(((gaussian_kde(sample))(hat) for sample,hat in zip(x_tmp,xi_hat_)),axis=0))
    f_hat=f_hat.reshape(x.shape[1:])
    se=np.sqrt(var_hat/n)/f_hat
  else:#method == "sub"
    a=n-b
    xi_hat=quant(g(x))
    y=np.stack([quant(g(x[k:k+b])) for k in range(a)],axis=0)
    mu_hat=np.mean(y,axis=0)
    var_hat=n*b*np.sum((y-mu_hat)**2,axis=0)/(a-1)/a
    se=np.sqrt(var_hat/n)
  return xi_hat, se

def mcse_p_mat(*args,**kwargs):
  """
  Only for mimicking MCMCSE.R, as the function mcse_p supports all numpy arrays.
  It thus only forwards its arguments to mcse_p.
  See mcse_p
  """
  return mcse_p(*args, **kwargs)

def mcse_q(x, q, *args, **kwargs):
  """
  Computes the MCMC estimate of quantile q, with standard error
  NOTE: function broadcasts over numpy arrays
  Args:
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
  Returns:
    * est: estimated quantile
        * est.shape == (n1,n2,...nk)
    * ess: estimated standard error of the quantile
        * ess.shape ==(n1,n2,...,nk)
    Note when they have dimension zero, numerics are returned    
  """
  assert 0<q<=1, "Quantile must be between 0 and 1"
  return mcse_p(x, q*100,*args,**kwargs)

def mcse_q_mat(*args,**kwargs):
  """
  Only for mimicking MCMCSE.R, as the function mcse_q supports all numpy arrays.
  It thus only forwards its arguments to mcse_q.
  See mcse_q
  """
  return mcse_q(*args, **kwargs)

def ess(x, g=None, **kwargs):
  """
  Estimate effective sample size (ESS) as described in Gong and Felgal (2015).
  Args:
    * x: data
        * if one sample is of shape (n1,n2, ..., nk)
          than x.shape == (n_samples, n1,n2,...nk)
    * g: function which expectation is to be computed and estimated
        * if a sample is of shape (n1,...,nk), g has to handle such inputs
    * kwargs: arguments to be passed to mcse
  Returns:
    * ess: estimated sample size
      * ess.shape == (n1,n2,...,nk)
  """
  g_x=g(x) if callable(g) else x
  n=len(g_x)
  lambda_=np.var(g_x,ddof=1,axis=0)
  _,sigma_=mcse(g_x, **kwargs)
  sigma=sigma_**2*n
  return n*lambda_/sigma
