import numpy as np
from scipy.io import mmwrite
import matplotlib.pyplot as plt
import sys
from pyfreefem import FreeFemRunner
from rompcm.paths import DATADIR, TESTSDIR


sys.path.append("..") 


_samples = None
_g1 = None
_samples_pars = None
_pod = None
_rbf = None
_Nt_pred = None
_dt = None
_Uscal=1.
_Vscal=1.


def init_worker(samples, g1, samples_pars, pod, rbf, Nt_pred, dt):
	global _samples, _g1, _samples_pars, _pod, _rbf, _Nt_pred, _dt
	_samples = samples
	_g1 = g1
	_samples_pars = samples_pars
	_pod = pod
	_rbf = rbf
	_Nt_pred = Nt_pred
	_dt = dt

def greedy_chunk(chunk):
	out = np.zeros((len(chunk), 4))  # [idx, error1, ...]

	for k, i in enumerate(chunk):
		theta = (_samples[i] - _pod.samples_mean) @ _pod.phi
		theta_r, out[k,-1] = _rbf.predict_theta(_samples_pars[i], _Nt_pred, theta)
		Vr = theta_r @ _pod.phi.T + _pod.samples_mean
		Vr += _samples_pars[i]*_g1
		out[k, 1:-1] = estimator_cond(Vr, _samples[i] +_samples_pars[i]*_g1, _samples_pars[i], _dt)
		out[k, 0] = i
	return out

def greedy_chunk_full(chunk):
	out = np.zeros((len(chunk), 4))  # [idx, error1, ...]

	for k, i in enumerate(chunk):
		V_r = []
		for j in range(3):
			theta = (_samples[j, i] - _pod[j].samples_mean) @ _pod[j].phi
			theta_r, out[k,-1]= _rbf[j].predict_theta(_samples_pars[i], _Nt_pred, theta)
			V_r.append(theta_r @ _pod[j].phi.T + _pod[j].samples_mean)
		V_r[-1] += _samples_pars[i]*_g1
		
		out[k, 1:-1] = estimator_conv_temperature(
												V_r[2], _samples[2,i]+_samples_pars[i]*_g1, 
												V_r[0]*_Uscal, V_r[1]*_Vscal, _samples_pars[i], _dt
												)
		out[k, 0] = i
	return out



def plot_snapshots(data1, data2, test_par, ref, label):
	x, y = np.loadtxt(DATADIR / "femcoordsp2nh60.txt", unpack=True)

	lev = 200
	fig, ax = plt.subplots(1, 3, figsize=(11.52, 3.12))

	ax[0].set_title(f'{ref} ', fontsize = 12)
	ax[0].set_ylabel(f'$\mu $ = {test_par}', fontsize = 12)
	ax[0].set_aspect('equal')
	a = ax[0].tricontourf(x, y, data1[-1], levels=lev, cmap="jet")
	fig.colorbar(a) 

	ax[1].set_title(f'{label}', fontsize = 12)
	ax[1].set_aspect('equal')
	b = ax[1].tricontourf(x, y, data2[-1], levels=lev, cmap="jet")
	fig.colorbar(b) 

	ax[-1].set_title(f'difference', fontsize = 12)
	ax[-1].set_aspect('equal')
	c = ax[-1].tricontourf(x, y, data1[-1] - data2[-1], levels=lev, cmap="jet")
	fig.colorbar(c) 

	plt.tight_layout()
	plt.show()
	

def estimator_cond(T_r, T_h, beta0, dt):

	residualT = f"""
		IMPORT "io.edp"

		load "medit"
		load "bfstream"
		real beta = {beta0};
		real dtt = {dt};

		include "{TESTSDIR}/Macro_operator.idp"
		include "{TESTSDIR}/param_phys.inc"

		mesh Mh = ThBackup;
		fespace Vh(Mh, P2);   

		Vh T0h, T1h, T2h, Thfh;
		
		// varf vMassBC(u, v) = int2d(Mh, qforder=6)(u * v) + on(4, u = 0.0);
		// matrix Mbc = vMassBC(Vh, Vh);

		T0h[] = importArray("T0"); T1h[] = importArray("T1"); 
		T2h[] = importArray("T2"); Thfh[] = importArray("Thf");

		real cc1=1.5/dtt, cc2=-2./dtt, cc3=0.5/dtt;

		Vh rt, T, ST2 = S(T2h), ST1 = S(T1h), ST0 = S(T0h);
		problem residual(rt, T) = int2d(Mh, qforder=6)(rt*T)
								- int2d(Mh, qforder=6)( cc1*(T2h + ST2)*T 
														+ grad(T2h)'*grad(T)*IPr 
														+ cc2*(T1h + ST1)*T 
														+ cc3*(T0h + ST0)*T)
								+ on(4, rt= 0)    // + on(4, rt=T2h - beta)
								;

		residual;
		real resh = int2d(Mh,qforder=6)( rt*rt );
		real err = int2d(Mh,qforder=6)((T2h - Thfh)*(T2h - Thfh)) / int2d(Mh,qforder=6)(Thfh*Thfh);
		// real  normgradT = int2d(Mh,qforder=6)(S(T2h)*T2h);
		// cout << " (S(T1), T1) " << normgradT << endl;
		exportVar(resh); 
		exportVar(err);
		// exportMatrix(Mbc);
	"""
	
	runner1 = FreeFemRunner(residualT)
	err_l2 = 0.; res_h = 0.
	Nt=0
	for i in range(2, T_r.shape[0]):
		runner1.import_variables(beta0=beta0, dtt=dt, T0=T_r[i-2],T1=T_r[i-1], T2=T_r[i], Thf=T_h[i])

		exports = runner1.execute(verbosity=0)
		res_h  += exports["resh"]
		err_l2 += exports["err"]
		Nt+=1
	#Mbc = exports["Mbc"]
	#mmwrite(TESTSDIR / "Mbc.mtx", Mbc)
	#assert False
	return (err_l2/Nt)**0.5, (res_h/Nt)**0.5



def estimator_conv_temperature(T_r, T_h, u_h, v_h, beta0, dt):

	residualT = f"""
		IMPORT "io.edp"
		real beta = {beta0};
		real dtt = {dt};
		include "{TESTSDIR}/Macro_operator.idp"
		include "{TESTSDIR}/param_phys.inc"

		real cc1=1.5/dtt, cc2=-2./dtt, cc3=0.5/dtt;

		mesh Mh = ThBackup;
		fespace Vh(Mh, P2);
		Vh u2h, v2h, T0h, T1h, T2h, Thfh; 

		u2h[] = importArray("u2"); v2h[] = importArray("v2");
		T0h[] = importArray("T0"); T1h[] = importArray("T1"); 
		T2h[] = importArray("T2"); Thfh[] = importArray("Thf");

		Vh rt, T;							
		problem residualT(rt, T) = int2d(Mh, qforder=6)(rt*T)
								- int2d(Mh, qforder=6)( cc1*(T2h + S(T2h))*T 
														+ (u2h*dx(T2h) + v2h*dy(T2h))*T
														+ grad(T2h)'*grad(T)*IPr 
														+ cc2*(T1h + S(T1h))*T 
														+ cc3*(T0h + S(T0h))*T)
								+ on(4, rt=T2h-beta);
		residualT;
		
		real resh = int2d(Mh,qforder=6)( rt*rt);
		real err = int2d(Mh,qforder=6)((T2h - Thfh)*(T2h - Thfh)) / int2d(Mh,qforder=6)(Thfh*Thfh);
		exportVar(resh); 
		exportVar(err);
	"""
	
	runner1 = FreeFemRunner(residualT)
	err_l2 = 0.; res_h = 0.
	N_t = 0
	for i in range(2, T_r.shape[0]):
		runner1.import_variables(beta0=beta0, dtt=dt, u2=u_h[i], v2=v_h[i], 
								 T0=T_r[i-2],T1=T_r[i-1], T2=T_r[i], Thf=T_h[i])

		exports = runner1.execute(verbosity=0)

		res_h += exports["resh"]
		err_l2 += exports["err"]
		N_t += 1

	return (err_l2 / N_t)**0.5, (res_h / N_t)**0.5



def estimator_conv_full(U_r, U_h, P_r, P_h, beta0, dt):

	residualT = r"""
		IMPORT "io.edp"
		include "{TESTSDIR}/Macro_operator.idp"
		
		real beta = importVar("beta0");
		include "{TESTSDIR}/param_phys.inc"
		real dtt = importVar("dtt");
		real cc1=1.5/dtt, cc2=-2./dtt, cc3=0.5/dtt;
		real gamma = 0.;

		mesh Mh = ThBackup;
		fespace Vh(Mh, P2);
		fespace Wh(Mh, P1);

		Vh u2h, v2h, T2h, u2r, v2r, u1r, v1r, u0r, v0r, T0r, T1r, T2r; 
		Wh p2h, p2r;

		u2h[] = importArray("u2");  v2h[] = importArray("v2"); T2h[] = importArray("T2");
		u2r[] = importArray("u2r"); v2r[] = importArray("v2r"); T2r[] = importArray("T2r");
		u1r[] = importArray("u1r"); v1r[] = importArray("v1r"); T1r[] = importArray("T1r");
		u0r[] = importArray("u0r"); v0r[] = importArray("v0r"); T0r[] = importArray("T0r"); 
		p2h[] = importArray("p2h"); p2r[] = importArray("p2r");
		
		Wh rp, q;
		Vh rt, T, ru, rv, v1, v2;						
		problem residual([ru, rv, rt, rp], [v1, v2, T, q]) = int2d(Mh, qforder=6)([ru, rv, rt, rp]'*[v1, v2, T, q])
								- int2d(Mh, qforder=6)( 
														 cc1*[u2r,v2r,T2r]'*[v1,v2,T]
														+ IRe*(grad(u2r)'*grad(v1) + grad(v2r)'*grad(v2)) - (dx(v1) + dy(v2))*p2r 
														- Amushy(T2r)*[u2r,v2r]'*[v1,v2] - fB(T2r)*v2
														+ [u2r,v2r]'*grad(T2r)*T + grad(T2r)'*grad(T)*IPr 
														+ cc2*[u1r,v1r,T1r]'*[v1,v2,T] + cc3*[u0r,v0r,T0r]'*[v1,v2,T]
														+ (cc1*S(T2r) + cc2*S(T1r) + cc3*S(T0r))*T 
														+ ( dx(u2r) + dy(v2r))*q
													   )
								+ on(1,2,3,4, ru=0, rv=0) + on(4, rt=T2r-beta);

		residual;
		/*
		cout << "resiual P " << sqrt(int2d(Mh,qforder=6)( rp*rp )) <<endl;
		cout << "resiual T " << sqrt(int2d(Mh,qforder=6)( rt*rt )) <<endl;
		cout << "resiual U " << sqrt(int2d(Mh,qforder=6)( ru*ru )) <<endl;
		cout << "resiual V " << sqrt(int2d(Mh,qforder=6)( rv*rv )) <<endl;
		*/
		real reshp = sqrt(int2d(Mh,qforder=6)( rp*rp ));
		real resht = sqrt(int2d(Mh,qforder=6)( rt*rt ));  
		real reshu = sqrt(int2d(Mh,qforder=6)( ru*ru ));
		real reshv = sqrt(int2d(Mh,qforder=6)( rv*rv));
		
		real errt =  sqrt( int2d(Mh,qforder=6)((T2h - T2r)*(T2h - T2r)) / int2d(Mh,qforder=6)(T2h*T2h) );
		real erru =  sqrt( int2d(Mh,qforder=6)((u2h - u2r)*(u2h - u2r)) / int2d(Mh,qforder=6)(v2h*v2h) ); 
		real errv =  sqrt( int2d(Mh,qforder=6)((v2h - v2r)*(v2h - v2r)) / int2d(Mh,qforder=6)(u2h*u2h) ); 
		real errp =  sqrt( int2d(Mh,qforder=6)((p2h - p2r)*(p2h - p2r)) / int2d(Mh,qforder=6)(p2h*p2h) );
		
		exportVar(reshp); 
		exportVar(resht); 
		exportVar(reshu); 
		exportVar(reshv); 
		exportVar(errt);
		exportVar(erru);
		exportVar(errv);
		exportVar(errp);
	"""
	
	runner1 = FreeFemRunner(residualT)
	err_t = 0.; err_u=0.; err_v=0.; err_p=0.; res_p = 0.; res_t = 0.; res_u = 0.; res_v = 0.
	N_t = 0
	for i in range(2, U_r[0].shape[0]):
		runner1.import_variables(beta0=beta0, dtt=dt, 
								 u2=U_h[0][i], v2=U_h[1][i], T2=U_h[2][i],
								 u2r=U_r[0][i], v2r=U_r[1][i], T2r=U_r[2][i],
								 u1r=U_r[0][i-1], v1r=U_r[1][i-1], T1r=U_r[2][i-1],
								 u0r=U_r[0][i-2], v0r=U_r[1][i-2], T0r=U_r[2][i-2],
								 p2r=P_r[i], p2h=P_h[i])

		exports = runner1.execute(verbosity=0)

		res_u += (exports["reshu"])**2; res_v += (exports["reshv"])**2; res_t += (exports["resht"])**2
		err_u += (exports["erru"])**2; err_v += (exports["errv"])**2;   err_t += (exports["errt"])**2
		err_p += (exports["errp"])**2; res_p += (exports["reshp"])**2
		N_t += 1
	
	return np.array([[(err_u / N_t)**0.5, (res_u / N_t)**0.5], 
					 [(err_v / N_t)**0.5, (res_v / N_t)**0.5],  
					 [(err_t / N_t)**0.5, (res_t / N_t)**0.5],
					 [(err_p / N_t)**0.5, (res_p / N_t)**0.5]])
