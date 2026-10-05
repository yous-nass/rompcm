import numpy as np
import pysindy as ps


class POD_Sindy:
	
	def __init__(self, 
	            theta:np.ndarray, 
      		    samples_µ:np.array,
		        dt:float=0.5,
		        eps:float=0.001, 
		        alpha:float=1.e-5, 
		        degree:int=1,
		        CONST:float=1.e-3,
		        normalized:str=True
		        ):

		ns, nt, nr = theta.shape
		x = [np.zeros((nt, nr+1)) for _ in range(ns)]
		for i, mu in enumerate(samples_µ):
			for k in range(nt):
				x[i][k, :-1] = theta[i, k]
				x[i][k, -1]  = CONST*mu
		x_train = x

		optimizer = ps.STLSQ(threshold=eps, alpha=alpha, normalize_columns=normalized)
		library = ps.PolynomialLibrary(degree=degree)
		self.sdy = ps.SINDy(optimizer=optimizer,feature_library=library,discrete_time=True)
		self.sdy.fit(x_train, t=[dt]*ns)
		self.CONST = CONST
	
	def predict_theta(self, theta0:np.array, Nt_pred:int=1, mu0:float=0.):
		return self.sdy.simulate(np.concatenate((theta0,[mu0*self.CONST])), Nt_pred)[:,:-1]
