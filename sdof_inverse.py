import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.special import gamma
import math
import time
#from google.colab import drive
import imageio


def fraccaputo(yt, h, a, tau):
    n = len(yt)
    frac_order=yt.clone()
    for k in range(1,n):
      memory = 0.0
      mem = [0, 0]
      mem[0] = max(0, k - math.ceil(tau / h))
      mem[1] = k
      k_st = mem[1] - mem[0]
      w = np.zeros(k_st)
      for j in range(k_st):
        if j == k_st - 1:
            w[k_st - 1 - j] = (j + 2) ** (1 - a) - 3 * (j + 1) ** (1 - a) + 2 * (j) ** (1 - a)
        else:
            w[k_st - 1 - j] = (j + 2) ** (1 - a) - 2 * (j + 1) ** (1 - a) + (j) ** (1 - a)
        memory += yt[k - 1 - j] * w[k_st - 1 - j]
      frac_order[k] = (h ** (-a) / gamma(2 - a)) * (yt[k - 1] + memory)
    return frac_order

def fraccaputo_v2(yt, h, a, tau,k):
    n = len(yt)
    frac_order=0.0
    memory = 0.0
    mem = [0, 0]
    mem[0] = max(0, k - math.ceil(tau / h))
    mem[1] = k
    k_st = mem[1] - mem[0]
    w = np.zeros(k_st)
    for j in range(k_st):
      if j == k_st - 1:
        w[k_st - 1 - j] = (j + 2) ** (1 - a) - 3 * (j + 1) ** (1 - a) + 2 * (j) ** (1 - a)
      else:
        w[k_st - 1 - j] = (j + 2) ** (1 - a) - 2 * (j + 1) ** (1 -a) + (j) ** (1 - a)
      memory += yt[k - 1 - j] * w[k_st - 1 - j]
    frac_order = (h ** (-a) / gamma(2 - a)) * (yt[k - 1] + memory)
    return frac_order

def fraccaputo_V3(yt,h,a,tau,k):
    device=yt.device
    st=time.time()
    frac_order=torch.zeros(yt.shape[0],1)
    memory=torch.zeros(yt.shape[0],1)
    mem=torch.tensor([0,0])
    mem[0]=torch.max(torch.tensor(0),k-torch.ceil(tau/h))
    mem[1]=k
    k_st=mem[1]-mem[0]
    j=torch.arange(k_st,device=device)
    j = torch.flip(j, dims=[0])
    w=torch.zeros_like(j,dtype=yt.dtype, device=device)
    w[0]=(j[0] + 2) ** (1 - a) - 3 * (j[0]+1) ** (1 - a) + 2*j[0] ** (1 - a)
    w[1:-1]=(j[1:-1] +1) ** (1 - a) - 2 * (j[1:-1] ) ** (1 - a) + (j[1:-1]-1) ** (1 - a)
    w[-1]=1
    fracorder=(h ** (-a) / torch.exp(torch.lgamma(2 - a)))*torch.sum(yt[-k_st:]*w)
    ed=time.time()
    ittim=ed-st
    return fracorder

def fraccaputo_V4(yt,h,a,tau):
    device=yt.device
    st=time.time()
    ytlen=yt.shape[0]
    frac_order=torch.zeros(ytlen,1)
    w=fracweights(a,tau,h,ytlen)
    fracorder=torch.matmul(w,yt)
    ed=time.time()
    ittim=ed-st
    return fracorder

def fracweights(a,tau,h,ytlen):
    w=torch.eye(ytlen)
    mem=torch.zeros(ytlen,1)
    k=torch.arange(ytlen)
    cols = torch.arange(ytlen).unsqueeze(0).expand(ytlen, -1)
    k_st=k - torch.max(torch.tensor(0),k-torch.ceil(tau/h))
    mask= (cols >= (k - k_st + 1).unsqueeze(1)) & (cols <= (k - 1).unsqueeze(1))
    j = (k.unsqueeze(1) - cols - 1)[mask]
    w[k.long(),(k-k_st).long()]=(k_st + 2) ** (1 - a) - 3 * (k_st+1) ** (1 - a) + 2*k_st ** (1 - a)
    w[mask]=((j+2+1e-6) ** (1 - a ) - 2 * (j+1+1e-6) ** (1 - a ) + (j+1e-6) ** (1 - a ))
    w[0,0]=0
    w=(h ** (-a) / torch.exp(torch.lgamma(2 - a))) *  w
    return w

def fraccaputo_V5(yt, h, a, tau):
    device = yt.device
    ytlen = yt.shape[0]

    def fracorder_mine(a, tau, h, ytlen, yt):
      w=torch.eye(ytlen)
      k=torch.arange(ytlen)
      cols = torch.arange(ytlen).unsqueeze(0).expand(ytlen, -1)
      k_st = k - torch.max(torch.tensor(0),k-torch.ceil(tau/h))
      mask = (cols >= (k - k_st + 1).unsqueeze(1)) & (cols <= (k -1).unsqueeze(1))
      j=torch.zeros(ytlen, ytlen)
      j = (k.unsqueeze(1) - cols -1)[mask]
      kk = k.long()[1:]
      kk_st = k_st.long()[1:]
      w[kk.long(),(kk-kk_st).long()]=(kk_st + 1) ** (1 - a) - 3* (kk_st) ** (1 - a) + 2*(kk_st - 1) ** (1 - a)
      w[mask]=((j+2) ** (1 - a ) - 2 * (j+1) ** (1 - a ) + (j) ** (1 - a ))
      w[0,0]=0
      w=(h ** (-a) / torch.exp(torch.lgamma(2 - a))) *  w
      fracorder=torch.matmul(w,yt)
      return fracorder

    # Create a differentiable wrapper for the weight calculation
    class WeightFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, tau, h, ytlen, yt):
            # w = fracweights(a, tau, h, ytlen)
            fracorder_forward = fracorder_mine(a, tau, h, ytlen, yt)
            ctx.save_for_backward(a, tau, h, torch.tensor(ytlen, device=device), yt)
            return fracorder_forward

        @staticmethod
        def backward(ctx, grad_output):
            a, tau, h, ytlen, yt = ctx.saved_tensors
            epsilon = 5*1e-1

            # Calculate numerical gradient for weights
            grad_fracorder_tau = torch.zeros_like(grad_output)
            fracorder_tau_1 = fracorder_mine(a, tau - epsilon, h, ytlen, yt)
            fracorder_tau_2 = fracorder_mine(a, tau + epsilon, h, ytlen, yt)
            grad_fracorder_tau = (fracorder_tau_2 - fracorder_tau_1) / (2*epsilon)
            
            # grad_fracorder_alpha = torch.zeros_like(grad_output)
            # fracorder_alpha_1 = fracorder_mine(a - epsilon, tau, h, ytlen, yt)
            # fracorder_alpha_2 = fracorder_mine(a + epsilon, tau, h, ytlen, yt)
            # grad_fracorder_alpha = (fracorder_alpha_2 - fracorder_alpha_1) / (2*epsilon)

            with torch.set_grad_enabled(True):
                fracorder_a = fracorder_mine(a, tau, h, ytlen, yt)
                grad_fracorder_alpha = torch.autograd.grad(fracorder_a, a, torch.ones_like(fracorder_a), create_graph=True)[0]

            # print(grad_alpha)
            # print(grad_fracorder.shape)
            # print(grad_alpha)

            # Apply chain rule to get gradient for alpha
            grad_fracorder_tau   = torch.sum(grad_output * grad_fracorder_tau)
            grad_fracorder_alpha = torch.sum(grad_output * grad_fracorder_alpha)

            # print(grad_fracorder.unsqueeze(0).shape)
            # print(grad_fracorder)

            return grad_fracorder_alpha.unsqueeze(0), grad_fracorder_tau.unsqueeze(0), None, None, None

    # Calculate weights using the differentiable wrapper
    # w = WeightFunction.apply(a, tau, h, ytlen)
    fracorder = WeightFunction.apply(a, tau, h, ytlen, yt)

    # Calculate the fractional derivative
    # fracorder = torch.matmul(w, yt)

    return fracorder

def smooth_ceil_exact(x):
    k=100
    x = torch.as_tensor(x, dtype=torch.float32)
    pi_x = math.pi * x
    transition = (2 / (1 + torch.exp(-k * torch.sin(pi_x)))) - 1
    smooth_value = x + (transition * torch.arcsin(torch.cos(pi_x))) / math.pi + 0.5
    return smooth_value

def smooth_floor_exact(x):
    k=100
    x = torch.as_tensor(x, dtype=torch.float32)
    pi_x = math.pi * x
    transition = (2 / (1 + torch.exp(-k * torch.sin(pi_x)))) - 1
    smooth_value = x + (transition * torch.arcsin(torch.cos(pi_x))) / math.pi - 0.5
    return smooth_value


# def fraccaputo_V6(yt, h, a, tau):
#       device = yt.device
#       ytlen = yt.shape[0]
#       w=torch.eye(ytlen)
#       k=torch.arange(ytlen, dtype=torch.int64)
#       k_st=torch.arange(ytlen)
#       cols = torch.arange(ytlen).unsqueeze(0).expand(ytlen, -1)
#     #   print(type(torch.max(torch.tensor(0),k-smooth_ceil_exact(tau/h))))
#       k_st = k - torch.max(torch.tensor(0),k-smooth_ceil_exact(tau/h))
#       mask= (cols >= (k - k_st + 1).unsqueeze(1)) & (cols <= (k -1).unsqueeze(1))
#       j=torch.zeros(ytlen, ytlen)
#       j = (k.unsqueeze(1) - cols -1)[mask]
#       kk = k[1:]
#       kk_st = k_st[1:]

#       ind = torch.complex(k_st[1:], 0).real()
#       print(type(ind))

#       w[kk,(kk-ind)]=(kk_st + 1) ** (1 - a) - 3* (kk_st) ** (1 - a) + 2*(kk_st - 1) ** (1 - a)
#       w[mask]=((j+2) ** (1 - a ) - 2 * (j+1) ** (1 - a ) + (j) ** (1 - a ))
#       w[0,0]=0
#       w=(h ** (-a) / torch.exp(torch.lgamma(2 - a))) *  w
#       fracorder=torch.matmul(w,yt)
#       return fracorder


def fraccaputo_fixed_tau(yt, h, a, tau_int):
    ytlen = yt.shape[0]
    w = torch.eye(ytlen, device=yt.device)
    k = torch.arange(ytlen, dtype=torch.int64, device=yt.device)

    k_st = k - torch.clamp(k - int(tau_int), min=0)

    cols = torch.arange(ytlen, dtype=torch.int64, device=yt.device).unsqueeze(0).expand(ytlen, -1)
    mask = (cols >= (k - k_st + 1).unsqueeze(1)) & (cols <= (k - 1).unsqueeze(1))
    j = (k.unsqueeze(1) - cols - 1)[mask]
    kk = k[1:]
    kk_st = k_st[1:]
    w[kk, (kk - kk_st)] = (kk_st + 1)**(1 - a) - 3 * (kk_st)**(1 - a) + 2 * (kk_st - 1)**(1 - a)
    w[mask] = (j + 2)**(1 - a) - 2 * (j + 1)**(1 - a) + j**(1 - a)
    w[0, 0] = 0
    w = (h ** (-a) / torch.exp(torch.lgamma(2 - a))) * w
    fracorder = torch.matmul(w, yt)
    return fracorder


def fraccaputo_V6_mine(yt, h, a, tau):
    tau_idx = tau / h
    tau_floor = torch.floor(tau_idx).detach()
    # tau_floor = smooth_floor_exact(tau_idx)
    tau_ceil = tau_floor + 1
    if tau_floor == T/h:
        tau_ceil = tau_floor-1 
    alpha = tau_idx - tau_floor
    alpha_mine = 1-alpha

    # minie = min(alpha, alpha_mine)
    # maxie = max(alpha, alpha_mine)

    if tau_actual - torch.floor(torch.Tensor([tau_actual])).detach() <= 0.5:
        minie = max(alpha, alpha_mine)
        maxie = min(alpha, alpha_mine)
    else:
        minie = min(alpha, alpha_mine)
        maxie = max(alpha, alpha_mine)

    tau_floor_val = tau_floor.item()
    tau_ceil_val = tau_ceil.item()
    frac1 = fraccaputo_fixed_tau(yt, h, a, tau_floor_val)
    frac2 = fraccaputo_fixed_tau(yt, h, a, tau_ceil_val)

    # Differentiable interpolation
    fracorder = minie * frac1 + maxie * frac2
    return fracorder


def fracSDOF(m,k,c,dt,F,x0,v0,T,a,tau):
   dt = float(dt)
   Nt = int(round(T/dt))
   u = np.zeros(Nt)
   t = np.linspace(0, T, Nt)
   u[0] = x0
   u[1] = u[0] + dt*v0 + (F[0] - k*u[0] - c*v0)/(2*m/dt**2)
   for n in range(1, Nt-1):
      frac=fraccaputo_v2( u[0:n+1], dt, a, tau, n )
      u[n+1]=2*u[n] - u[n-1] + (F[n]-k*u[n]-c*frac)/(m/dt**2)
   return u,t

def FracSDOF(m,k,c,dt,F,x0,v0,T,a,tau):
    dt = float(dt)
    Nt = int(np.ceil(T/dt))
    u = np.zeros(Nt)
    t = np.linspace(0, T, Nt)
    u[0] = x0
    u[1] = u[0] + dt*v0 + (F[0] - k*u[0] - c*v0)/(2*m/dt**2)
    tf = int(np.ceil(tau/dt))
    pv = np.arange(tf + 3)
    pv = np.power(pv, 1 - a)
    Weight_vector = (pv[2:] - 2 * pv[1:-1] + pv[:-2])[:tf]
    for n in range(1, Nt - 1):
        if (n < tf):
            wv = Weight_vector[:n].copy()
            wv[n - 1] = wv[n - 1] + pv[n - 1] - pv[n]
            wv = np.flip(wv,0)
            wv = np.concatenate([wv, np.array([1.])])
            wv = wv * (dt ** (-a) / math.gamma(2 - a))
            u[n + 1] = 2 * u[n] - u[n - 1] + (F[n] - k * u[n] - c * np.dot(wv, u[:n + 1])) / (m / dt ** 2)
        else:
            wv = Weight_vector.copy()
            wv[tf - 1] = wv[tf - 1] + pv[tf - 1] - pv[tf]
            wv = np.flip(wv,0)
            wv = np.concatenate([wv, np.array([1.])])
            wv = wv * (dt ** (-a) / math.gamma(2 - a))
            u[n + 1] = 2 * u[n] - u[n - 1] + (F[n] - k * u[n] - c * np.dot(wv, u[n - tf :n+1])) / (m / dt ** 2)
    return u, t

#---- Layer classes ----#

class Sine(nn.Module):
    def __init__(self, layer_size : list[int], freq):
        super().__init__()
        self.input_size, self.output_size = layer_size
        self.linear = nn.Linear(self.input_size,self.output_size)
        self.frequency = freq
    def forward(self, x):
        return torch.sin(self.frequency*self.linear(x))

#---- NN class ----#

class Net(nn.Module):
    def __init__(self,layer_size : list[int],layer_count, freq) -> None:
        super().__init__()
        self.input_size,self.hidden_size,self.output_size = layer_size
        activation = nn.Tanh

        self.Ext_Layers = [Sine([self.input_size,self.hidden_size],freq)]

        self.Layer = nn.ModuleList([nn.Linear(self.hidden_size,self.hidden_size)])

        for _ in range(layer_count - 1):
            self.Layer.extend([nn.Linear(self.hidden_size, self.hidden_size), activation()])
        
        self.Out_Layer = nn.Linear(self.hidden_size,self.output_size)

    def forward(self,x):
        # print(x.shape)
        for layers in self.Ext_Layers:
            x = layers(x)
        # print(x.shape)
        for layers in self.Layer:
            x = layers(x)
        return self.Out_Layer(x)

#drive.mount('/content/drive')
torch.manual_seed(123)
pinn=Net([1,32,1],3,10)
m,k,c=2,200,2
xo ,vo =  0.5,0.

a_actual=0.5
a1=torch.tensor([a_actual], requires_grad=True)

T=20
tau_actual=2
tau1 = torch.tensor([10.], requires_grad=True)

dt=0.01
N= float(T/dt)
t_bound1 = torch.tensor(0.,requires_grad=True).view(-1,1)
t_phy =torch.linspace(0,20,2000,requires_grad=True).view(-1,1)
t_test=torch.linspace(0,20,1000).view(-1,1)
t=np.arange(0,T,dt)
#F=np.ones_like(t)

F=np.sin(10*t)
u_fdm,t_fdm=FracSDOF(m,k,c,dt,F,xo,vo,T,a_actual,tau_actual)
num_indices=1000
index = np.linspace(0, t.shape[0] - 1, num_indices, dtype=int)
t_obs = torch.tensor(t[index], dtype=torch.float32).view(-1, 1)
u_obs = torch.tensor(u_fdm[index]).view(-1, 1) + 0.02 * torch.randn_like(t_obs)
images = []

# a_list=[]
tau_list=[]

iters=[]
l=[]
# a1=torch.nn.Parameter(torch.tensor(0.9),requires_grad=True)
optimiser     = torch.optim.Adam(list(pinn.parameters()), lr=1e-3)
optimiser_tau = torch.optim.Adam([tau1], lr=1e-1)
# optimiser_alpha = torch.optim.Adam([a1], lr=1e-3)

if a_actual == 0.25:
    mull = 3.9
elif a_actual== 0.75:
    mull = 5.9 
  
for i in range(15000):
    iters.append(i)
    st=time.time()      
    optimiser.zero_grad()
    optimiser_tau.zero_grad()
    # optimiser_alpha.zero_grad()
    lam1,lam2 = 1e-4, 1e-1

    #u=pinn(t_bound1)
    #print(u.size())
    #loss1=torch.squeeze((u - xo)**2)
    #dudt= torch.autograd.grad(u,t_bound1,torch.ones_like(u),create_graph=True)[0]
    #loss2=torch.squeeze((dudt - vo)**2)

    u=pinn(t_phy)
    F_phy=1*torch.sin(10*t_phy).view(-1,1)
    dudt= torch.autograd.grad(u,t_phy,torch.ones_like(u),create_graph=True)[0]
    fracst1=time.time()
    dudt_frac = fraccaputo_V6_mine(u,torch.tensor([0.01]),a1,tau1)
    fraced1=time.time()
    du2dt= torch.autograd.grad(dudt,t_phy,torch.ones_like(u),create_graph=True)[0]
    loss3=torch.mean((m*du2dt + k*u +c*dudt_frac - F_phy)**2)
    u1=pinn(t_obs)
    loss4=torch.mean((u1-u_obs)**2)
    loss=lam1*loss3+lam2*loss4
    l.append(loss.detach())
    loss.backward()
    optimiser.step()
    optimiser_tau.step()
    # optimiser_alpha.step()

    with torch.no_grad():
      a1.data = torch.clamp(a1.data, 0.1, 0.99)
      tau_list.append(tau1.item())
    #   a_list.append(a1.item())

    ed=time.time()
    iteration_time = ed - st
    frac_time=fraced1-fracst1
    #print(f"Training step {i}, Time taken: {iteration_time:.4f} s,'Frac compute time : {frac_time:.4f} s")
    if i % 39 == 0:
      print(f"Training step {i}, Time taken: {iteration_time:.4f} s,Frac compute time : {frac_time:.4f}, Total loss : {loss} s, alpha: {a1}, alpha_grad: {a1.grad}, tau: {tau1}, tau_grad: {tau1.grad}")
      u=pinn(t_test).detach()

      plt.figure(figsize=(15,20))
      plt.subplot(3,1,1)
      plt.plot(t_test[:,0],u[:,0])
      plt.plot(t_fdm,u_fdm)
      plt.legend(["PINN","FDM"])
      plt.xlabel('Time(s)')
      plt.ylabel('Displacement(mm)')
      plt.title('Output from PINN')

    #   plt.subplot(3,1,2)
    #   plt.plot(iters, a_list)
    #   plt.hlines(a_actual,0,len(a_list), colors=['g'])
    #   plt.legend(["PINN","Exact"])
    #   plt.xlabel('Iters')
    #   plt.ylabel('alpha')
    #   plt.title('Prediction of Fractional Order alpha')
    #   plt.suptitle(f"Training step {i}")

      plt.subplot(3,1,2)
      plt.plot(iters, tau_list)
      plt.hlines(tau_actual,0,len(tau_list), colors=['g'])
      plt.legend(["PINN","Exact"])
      plt.xlabel('Iters')
      plt.ylabel('tau')
      plt.title('Prediction of Fractional Order tau')
      plt.suptitle(f"Training step {i}")

      plt.subplot(3,1,3)
      plt.plot(iters, l)
    #   plt.hlines(tau_actual,0,len(tau_list), colors=['g'])
    #   plt.legend(["total","Exact"])
      plt.xlabel('Iters')
      plt.ylabel('Loss')
      plt.title('Loss Curve')
      plt.suptitle(f"Training step {i}")
      plt.savefig(f'Inverse_Final_V2_tau__{tau_actual}_alpha_{a_actual}.jpg')
      

      fig = plt.gcf()  # Get current figure
      fig.canvas.draw()  # Draw the canvas
      image_rgba = fig.canvas.buffer_rgba()
        # Get width and height from buffer shape
      width, height = fig.canvas.get_width_height()
      image_rgb = np.frombuffer(image_rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
      images.append(image_rgb)
      plt.close(fig)
      if i%78 == 0:
        imageio.mimsave(f'Inverse_animation_tau_{tau_actual}_alpha_{a_actual}.gif', images, fps=10)
      if loss.item()<mull*1e-5:
         torch.save({"model_parameters" : pinn.state_dict(), "inverse_parameters" : tau1.detach().cpu().numpy()}, f'tau_inverse_{tau_actual}_{a_actual}.pth')
         break
      if i%999 == 0:
         torch.save({"model_parameters" : pinn.state_dict(), "inverse_parameters" : tau1.detach().cpu().numpy()}, f'tau_inverse_{tau_actual}_{a_actual}.pth')
       
# torch.save(pinn.state_dict(), f'tau_inverse_{tau_actual}.pth')


