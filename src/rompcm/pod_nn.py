import numpy as np
import matplotlib.pyplot as plt
import torch as tr
import torch.nn as nn
from   typing import Tuple


################################################################
#  LSTM time forecasting model
################################################################
class LSTM(nn.Module):
	def __init__(self, 
				input_size=1, 
				hidden_size=64, 
				num_layers=1, 
				output_size=1,
				dropout=0,
				n_future=1
				):
		
		super(LSTM, self).__init__()
		
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.n_future = n_future
		
		self.lstm = nn.LSTM(input_size=input_size, 
							hidden_size=hidden_size,
							num_layers=num_layers, 
							dropout=dropout, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)
		
	def forward(self, x:tr.tensor):
		out, _ = self.lstm(x) 
		#out = out[:, -1, :] 
		out = self.fc(out)
		return out
	






################################################################
#  1d Fourier Integral Operator
################################################################
class SpectralConv1d(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, modes: int):
		super(SpectralConv1d, self).__init__()
		"""
		Initializes the 1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
		Args:
			in_channels (int): input channels to the FNO layer
			out_channels (int): output channels of the FNO layer
			modes (int): number of Fourier modes to multiply, at most floor(N/2) + 1
		"""
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.modes = modes
		self.scale = (1 / (in_channels*out_channels))
		self.weights = nn.Parameter(self.scale * tr.rand(in_channels, out_channels, self.modes, dtype=tr.cfloat))

	# Complex multiplication
	def compl_mul1d(self, input, weights):
		"""
		Complex multiplication of the Fourier modes.
		[batch, in_channels, x], [in_channel, out_channels, x] -> [batch, out_channels, x]
			Args:
				input (torch.Tensor): input tensor of size [batch, in_channels, x]
				weights (torch.Tensor): weight tensor of size [in_channels, out_channels, x]
			Returns:
				torch.Tensor: output tensor with shape [batch, out_channels, x]
		"""
		return tr.einsum("bix,iox->box", input, weights)

	def forward(self, x: tr.Tensor) -> tr.Tensor:
		"""
		Fourier transformation, multiplication of relevant Fourier modes, backtransformation
		Args:
			x (torch.Tensor): input to forward pass os shape [batch, in_channels, x]
		Returns:
			torch.Tensor: output of size [batch, out_channels, x]
		"""
		batchsize = x.shape[0]
		# Fourier transformation
		x_ft = tr.fft.rfft(x)

		# Multiply relevant Fourier modes
		out_ft = tr.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=tr.cfloat)
		out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

		#Return to physical space
		x = tr.fft.irfft(out_ft, n=x.size(-1))
		return x


 ################################################################
#  1d Fourier Network
################################################################
class FNO1d(nn.Module):
	def __init__(self, modes, width, time_future, time_history, activation):
		super(FNO1d, self).__init__()

		"""
		The overall network. It contains 4 layers of the Fourier layer.
		1. Lift the input to the desire channel dimension by self.fc0 .
		2. 4 layers of the integral operators u' = (W + K)(u).
			W defined by self.w; K defined by self.conv .
		3. Project from the channel space to the output space by self.fc1 and self.fc2 .

		input: a driving function observed at T timesteps + 1 locations (u(1, x), ..., u(T, x),  x).
		input shape: (batchsize, x=s, c=2)
		output: the solution of a later timestep
		output shape: (batchsize, x=s, c=1)
		"""
		
		self.modes = modes
		self.width = width
		self.time_future = time_future
		self.time_history = time_history
		self.fc0 = nn.Linear(self.time_history+1, self.width)

		self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
		self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
		self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
		self.conv3 = SpectralConv1d(self.width, self.width, self.modes)
		self.w0 = nn.Conv1d(self.width, self.width, 1)
		self.w1 = nn.Conv1d(self.width, self.width, 1)
		self.w2 = nn.Conv1d(self.width, self.width, 1)
		self.w3 = nn.Conv1d(self.width, self.width, 1)

		self.fc1 = nn.Linear(self.width, self.width)
		self.fc2 = nn.Linear(self.width, self.time_future)
		self.activation = activation
		
	def forward(self, u:tr.Tensor):
		
		grid = self.get_grid(u.shape, u.device)
		x = tr.cat((u.permute(0, 2, 1) , grid), dim=-1)
	
		x = self.fc0(x)
		x = x.permute(0, 2, 1) 

		x1 = self.conv0(x)
		x2 = self.w0(x)
		x = x1 + x2
		x = self.activation(x)

		x1 = self.conv1(x)
		x2 = self.w1(x)
		x = x1 + x2
		x = self.activation(x)

		x1 = self.conv2(x)
		x2 = self.w2(x)
		x = x1 + x2
		x = self.activation(x)

		x1 = self.conv3(x)
		x2 = self.w3(x)
		x = x1 + x2

		x = x.permute(0, 2, 1) 
		x = self.fc1(x)
		x = self.activation(x)
		x = self.fc2(x)
		return x.permute(0, 2, 1) # + u[:, -1:,:]
		
	def get_grid(self, shape, device):
		batchsize, size_x = shape[0], shape[-1]
		gridx = tr.tensor(np.linspace(0, 1, size_x), dtype=tr.float)
		gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
		return gridx.to(device)     
			


###############################################################################
###################### POD - NN ######################################### 
###############################################################################
def compute_pod(X, r):
	"""
	X: (N, T) snapshot matrix
	r: number of retained modes
	"""
	# SVD
	U, S, Vh = tr.linalg.svd(X, full_matrices=False)
	
	# Truncate
	Phi = U[:, :r]                      # (N, r)
	S_r = S[:r]
	Vh_r = Vh[:r, :]                    # (r, T)
	
	# POD coefficients
	alpha = tr.diag(S_r) @ Vh_r      # (r, T)
	
	return Phi, alpha



class POD:

	def __init__(self, eps:float=1.e-7, atol:float=1.e-8):
		self.eps = eps
		self.atol = atol

	def fit(self, samples):
		assert len(samples.shape) ==3, "the size of samples must be at least 3: (parameter, time, space)"
		ns, nt, nh = samples.shape
		
		samples_ = samples.reshape(ns*nt, nh)
		self.samples_mean = np.mean(samples_, 0, keepdims=True) 
		samples_ = samples_ - self.samples_mean
		# SVD
		U, S, _ = np.linalg.svd(samples_, full_matrices=False)

		# Truncate
		cum_energy = np.cumsum(S**2) /  np.sum(S**2)
		r = int(np.searchsorted(cum_energy, 1 - self.eps)) + 1
		
		Phi = samples_.T @ U[:, :r] / S[None,:r] 
		
		err = np.linalg.norm(Phi.T @ Phi - np.eye(r))
		
		assert err <= self.atol, f"orthogonality error = {err}"
		# POD coefficient of shape (ns, nt, nr)	
		self.Phi = Phi
		alpha = samples_ @ self.Phi
		self.alpha = alpha.reshape(ns, nt, r)
		
		return self



class POD_LSTM(nn.Module):
	def __init__(self, r, hidden_dim=64, num_layers=2, dropout=0):
		super().__init__()
		
		self.lstm = nn.LSTM(r, hidden_dim, num_layers, batch_first=True, dropout=dropout)
		self.fc = nn.Linear(hidden_dim, r)
	
	def forward(self, x):
    	# x: (batch, 2, r)
		out, _ = self.lstm(x)          # (batch, 2, hidden_dim)
		
		out_last = out[:, -1, :]       # (batch, hidden_dim)
		
		alpha_next = self.fc(out_last) # (batch, r)
		
		return alpha_next[:,None,:] + tr.mean(x, 1)[:,None,:]