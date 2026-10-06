import pickle
import numpy as np
import os, sys

from rompcm.paths import podcond, podconv, rbfcond, rbfconv
from rompcm.tools import estimator_cond, estimator_conv_temperature


class PODRBFInference:
    def __init__(self, dircase:str="cond"):
        self.dircase = dircase

        if self.dircase == "cond":
            pod_path = podcond
            rbf_path = rbfcond

            with open(pod_path, "rb") as f:
                model = pickle.load(f)
                self.pod = model["pod"]
            self.g1 = model["metadata"]["g1"]
            self.tol = model['metadata']["TOL"]
            self.dt = model['metadata']["dt"]

            with open(rbf_path, "rb") as f:
                model = pickle.load(f)
                self.rbf = model["rbf"]

        else:
            pod_path = podconv
            rbf_path = rbfconv

            with open(pod_path, "rb") as f:
                model = pickle.load(f)
                self.pod = model["pod"]
            self.g1 = model["metadata"]["g1"]
            self.tol = model['metadata']["TOL"]
            self.dt = model['metadata']["dt"]

            with open(rbf_path, "rb") as f:
                model = pickle.load(f)
                self.rbf = model["rbf"]

        

    def predict(self, mu, Nt):
        pred = []
        if self.dircase == "cond":
            # Predict reduced coefficients
            theta_pred, _ = self.rbf.predict_theta(mu, Nt)
            # Reconstruct full field
            pred.append(theta_pred @ self.pod.phi.T)
            pred[-1] += self.pod.samples_mean
            pred[-1] += mu*self.g1
            _, residual_error = estimator_cond(pred[-1], pred[-1], mu, self.dt)
        else:
            # Predict reduced coefficients
            
            for j in range(3):
                theta_r, _ = self.rbf[j].predict_theta(mu, Nt)
                pred.append(theta_r @ self.pod[j].phi.T + self.pod[j].samples_mean)
            pred[-1] += mu*self.g1
            _, residual_error = estimator_conv_temperature(
											pred[-1], pred[-1],
											pred[0], pred[1], mu, self.dt
											)

        if residual_error > self.tol:
            print("Warning: ROM uncertainty high, residual error:: ", residual_error)
        return pred[-1]
    

    def predict_field_point(self, mu0, t0):
        """
        Compute θ(mu0, t0) using stored RBF model
        """

        # Query point
        query = np.array([mu0, t0])   # (2,)

        if self.dircase == "cond":
            r = np.linalg.norm(self.rbf.mu_t - query, axis=1)  # (Ntrain,)
            K_q = self.rbf._kernel_eval(r)    
            theta_point = K_q @ self.rbf.coeffs   # (nr,)

            field = theta_point @ self.pod.phi.T
            field += np.squeeze(self.pod.samples_mean + mu0*self.g1)
        else:
            r = np.linalg.norm(self.rbf[-1].mu_t - query, axis=1)  # (Ntrain,)
            K_q = self.rbf[-1]._kernel_eval(r)    
            theta_point = K_q @ self.rbf[-1].coeffs   # (nr,)
            field = theta_point @ self.pod[-1].phi.T
            field += np.squeeze(self.pod[-1].samples_mean + mu0*self.g1)
        return field


