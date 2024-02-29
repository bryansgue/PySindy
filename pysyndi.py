import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp
from scipy.signal import lfilter
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from scipy.io import loadmat

# Cargar el archivo .mat
u_ref2= loadmat('u_ref_3.mat')
data2 = loadmat('states_3.mat')
time2 = loadmat('t_3.mat')
ext2 = 660
states2 = data2['states']
u2 = u_ref2['u_ref']
v_body2 = states2[19:22, 0:ext2]
euler_p2 = states2[9:12, 0:ext2]
omega2 = euler_p2[2,:]
# Asegúrate de que omega sea una matriz de una sola fila
omega2 = omega2.reshape(1, -1)
# Apilar verticalmente v y omega
v2 = np.vstack((v_body2, omega2))
v2 = v2.T

u_train2 = u2[:,0:ext2]
u_train2 = u_train2.T

## SEGUNDOS DATOS
# Cargar el archivo .mat
u_ref3= loadmat('u_ref_5.mat')
data3 = loadmat('states_5.mat')
time3 = loadmat('t_5.mat')
ext3 = 660
states3 = data3['states']
u3 = u_ref3['u_ref']
v_body3 = states3[19:22, 0:ext3]
euler_p3 = states3[9:12, 0:ext3]
omega3 = euler_p3[2,:]
# Asegúrate de que omega sea una matriz de una sola fila
omega3 = omega3.reshape(1, -1)
# Apilar verticalmente v y omega
v3 = np.vstack((v_body3, omega3))
v3 = v3.T

u_train3 = u3[:,0:ext3]
u_train3 = u_train3.T



#### TERCERODATOS

# Cargar el archivo .mat
u_ref= loadmat('u_ref_4.mat')
data = loadmat('states_4.mat')
time = loadmat('t_4.mat')


ext = 660


# Acceder a la variable 'states'
states = data['states']


t = time['t']
t_vector = t[0,0:ext]
dt = 1/30

u = u_ref['u_ref']


# Asignar las variables según los índices especificados
#v = states[0:3, 0:100]
#v = states[3:6, 0:100]
#euler = states[6:9, 0:100]
euler_p = states[9:12, 0:ext]
#v = states[12:15, 0:10]
quat = states[15:19, :]
v_body = states[19:22, 0:ext]

omega = euler_p[2,:]

# Asegúrate de que omega sea una matriz de una sola fila
omega = omega.reshape(1, -1)

# Apilar verticalmente v y omega
v1 = np.vstack((v_body, omega))

v1 = v1.T







u_train = u[:,0:ext]

u_train = u_train.T


# Fit the model



library1 = ps.PolynomialLibrary(degree=1)
library2 = ps.FourierLibrary(n_frequencies=1)
lib_generalized = ps.GeneralizedLibrary([library1, library2])
lib_generalized.fit(v1)


x_train_multi = []
u_train_multi = []

# Agrega los vectores a la lista x_train_multi
x_train_multi.append(v1)
x_train_multi.append(v1)
x_train_multi.append(v1)
x_train_multi.append(v1)

# Agrega los vectores a la lista x_train_multi
u_train_multi.append(u_train)
u_train_multi.append(u_train)
u_train_multi.append(u_train)
u_train_multi.append(u_train)


model = ps.SINDy(
    
    optimizer=ps.STLSQ(threshold=0.01, alpha=.05),
    feature_library=library1,
    differentiation_method = ps.SINDyDerivative(kind="kalman", alpha=0.05),
    feature_names=["v_l", "v_m","v_n","v_w", "u_l", "u_m","u_n","u_w" ],
    
)
model.fit(x_train_multi, u = u_train_multi, t=dt, multiple_trajectories=True)
model.print()



x0_test = np.array([0, 0, 0, 0])



x_test_sim = model.simulate(x0=x0_test, t=t_vector, u=u_train )

# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(v1, u=u_train)  

# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(v1, t=dt)


# Plot original data and model prediction
plt.figure(figsize=(10, 6))


# Plot original data
plt.subplot(2, 2, 1)
plt.plot(range(len(x_dot_test_computed[:, 0])), x_dot_test_computed[:, 0], label='Original Data', color='blue')
plt.plot(range(len(x_dot_test_predicted [:, 0])), x_dot_test_predicted [:, 0], label='Model Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 0]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot second variable
plt.subplot(2, 2, 2)
plt.plot(range(len(x_dot_test_computed[:, 1])), x_dot_test_computed[:, 1], label='Original Data', color='blue')
plt.plot(range(len(x_dot_test_predicted [:, 1])), x_dot_test_predicted [:, 1], label='Model Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 1]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot third variable
plt.subplot(2, 2, 3)
plt.plot(range(len(x_dot_test_computed[:, 2])), x_dot_test_computed[:, 2], label='Original Data', color='blue')
plt.plot(range(len(x_dot_test_predicted [:, 2])), x_dot_test_predicted [:, 2], label='Model Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 2]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot fourth variable
plt.subplot(2, 2, 4)
plt.plot(range(len(x_dot_test_computed[:, 3])), x_dot_test_computed[:, 3], label='Original Data', color='blue')
plt.plot(range(len(x_dot_test_predicted [:, 3])), x_dot_test_predicted [:, 3], label='Model Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 3]) and Model Prediction')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()




# Plot original data and model prediction
plt.figure(figsize=(10, 6))


# Plot original data
plt.subplot(2, 2, 1)
plt.plot(range(len(x_test_sim[:, 0])), x_test_sim[:, 0], label='Prediction Data', color='blue')
plt.plot(range(len(v1[:, 0])), v1[:, 0], label='Model Real', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 0]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot second variable
plt.subplot(2, 2, 2)
plt.plot(range(len(x_test_sim[:, 1])), x_test_sim[:, 1], label='Prediction Data', color='blue')
plt.plot(range(len(v1[:, 1])), v1[:, 1], label='Model Real', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 1]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot third variable
plt.subplot(2, 2, 3)
plt.plot(range(len(x_test_sim[:, 2])), x_test_sim[:, 2], label='Prediction Data', color='blue')
plt.plot(range(len(v1[:, 2])), v1[:, 2], label='Model Real', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 2]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot fourth variable
plt.subplot(2, 2, 4)
plt.plot(range(len(x_test_sim[:, 3])), x_test_sim[:, 3], label='Prediction Data', color='blue')
plt.plot(range(len(v1[:, 3])), v1[:, 3], label='Model Real', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 3]) and Model Prediction')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()