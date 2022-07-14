import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from Ball_Parameters import Ball_Robot


Kp, Kd = np.array([[1,0],[0,1]]) , np.array([[10,0],[0,10]])
KI = np.array([[.01,0],
              [0, .01]])

dt = 0.02 # 200 Hz
end_time = 10
t = np.arange(0,end_time, dt)

robot = Ball_Robot()
drive = robot.desired_drive
steer = robot.desired_steer
eint = np.zeros((2,2))

def model(q, t): # q = [th1, th2, thd1, thd2]
    # read desired path
    qd2, qdotd2, accel_des1 = steer(t)
    qd1, qdotd1, accel_des2 = drive(t, end_time)

    # get dynamics
    M = robot.get_M_static(q)
    G = robot.get_V_static(q)

    # set up error and path vectors
    e = np.array([[q[0] - qd1],
                  [q[1] - qd2]])
    edot = np.array([[q[2] - qdotd1],
                     [q[3] - qdotd2]])
    accel_des = np.array([[accel_des1],
                          [accel_des2]])
    # calculate torque
    T = M @ (-Kp @ e - Kd @ edot -accel_des) + G

    # command the torque
    accel_vec = np.linalg.inv(M) @ (T - G)
    return q[2], q[3], accel_vec[0][0], accel_vec[1][0]

# integrate the models

sol = odeint(model, [0.0, 0.0, 0.0, 0.0], t)

dq2dt = np.zeros((len(t), 2))
for i in range(len(t)):
    out = model(sol[i,:], t[i])
    dq2dt[i,0], dq2dt[i,1] = out[2], out[3]


#plots
# plt.figure(figsize=(6,7))
# plt.subplot(3,1,1)
plt.figure(1)
plt.title('Controlled Torque Response of a Static Pendulum')
plt.plot(t, sol[:,0], label = r'$\theta_1$')
plt.plot(t, sol[:,1], label = r'$\theta_2$')
steered_plot = np.zeros(len(t))
drive_plot = np.zeros(len(t))
for i in range(len(t)):
    steered_plot[i] = steer(t[i])[0]
    drive_plot[i] = drive(t[i], end_time)[0]
plt.plot(t, steered_plot, 'C1--',  label=r'$Desired \theta_2$')
plt.plot(t, drive_plot, 'C0--', label=r'$Desired Drive \theta_1$')
plt.grid()
plt.legend()
plt.figure(2)
plt.title('Velocity Response')
plt.plot(t, sol[:,2], label = r'$\dot{\theta}_1$')
plt.plot(t, sol[:,3], label = r'$\dot{\theta}_2$')
steered_plot = np.zeros(len(t))
drive_plot = np.zeros(len(t))
for i in range(len(t)):
    steered_plot[i] = steer(t[i])[1]
    drive_plot[i] = drive(t[i], end_time)[1]
plt.plot(t, steered_plot, 'C1--',  label=r'$Desired \dot{\theta}_2$')
plt.plot(t, drive_plot, 'C0--', label=r'$Desired Drive \dot{\theta}_1$')
plt.grid()
plt.legend()

plt.figure(3)
plt.title('Applied Torques')
tao = np.zeros((2, len(dq2dt[:,0])))

for i in range(len(dq2dt[:,0])):
    tao[:,i] = robot.get_M_static(0) @ dq2dt[i,:] + np.transpose(robot.get_V_static(sol[i,0:2]))[0]
plt.plot(t, tao[0,:], label = r'$\tau_1$')
plt.plot(t, tao[1,:], label = r'$\tau_2$')
steered_plot = np.zeros(len(t))
drive_plot = np.zeros(len(t))
# for i in range(len(t)):
#     steered_plot[i] = steer(t[i])[0]
#     drive_plot[i] = drive(t[i], end_time)[0]
# plt.plot(t, steered_plot, 'C1--',  label=r'$Desired \theta_2$')
# plt.plot(t, drive_plot, 'C0--', label=r'$Desired Drive \theta_1$')
plt.grid()
plt.legend()
plt.show()
