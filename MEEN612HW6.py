import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


m1 = 5
m2 = 5
L1 = 1
L2 = 1
l1 = L1/2
l2 = L2/2
g = 9.81

Iyy1, Iyy2 = 10,10
Izz, Ixx2 = 10,10
Kp, Kd = np.array([[10,0],[0,10]]) , np.array([[10,0],[0,10]])
lamda = 0.1
t = np.linspace(0,100, 100)

def equ1(q, t, Kp,Kd):
    t1, t2, t1d, t2d = q
    # the controlling variables Part 1
    #T = -Kp*np.array([[phi], [theta]])-Kd*np.array([[phidot], [thetadot]])

    # the first equation
    dq1dt = np.array([[t1d], [t2d]])
    q1 = np.array([[t1], [t2]])
    # the matricies
    M11 = m1*l1**2 + m2*(L1+l2*np.cos(t2))**2+ Iyy1+Iyy2*np.cos(t2)**2 + Ixx2*np.sin(t2)**2
    M12 = 0
    M21 = 0
    M22 = m2*l2**2 + Izz
    M = np.array([[M11, M12], [M21, M22]])

    C11 = -m2*(L1+l2*np.cos(t2))*l2*np.sin(t2)*t2d + (Ixx2-Iyy2)*np.sin(t2)*np.cos(t2)*t2d
    C12 = -m2*(L1+l2*np.cos(t2))*l2*np.sin(t2)*t1d+ (Ixx2-Iyy2)*np.sin(t2)*np.cos(t2)*t1d
    C21 = m2*(L1+l2*np.cos(t2))*l2*np.sin(t2)*t1d+ (Iyy2-Ixx2)*np.sin(t2)*np.cos(t2)*t1d
    C22 = 0
    C = np.array([[C11, C12], [C21, C22]])

    G = np.array([[0], [m2*g*L2*np.cos(t2)]])
    # Desired Path variables
    t1_des = 0.5*(1-np.cos(np.pi*t/60))
    t2_des = 0.5*(1+np.cos(np.pi*t/20))
    qDes = np.array([[t1_des], [t2_des]])
    t1d_des = 0.5*np.sin(np.pi*t/60)*np.pi/60
    t2d_des = -0.5*np.sin(np.pi*t/20)*np.pi/20
    qdDes = np.array([[t1d_des], [t2d_des]])
    t1dd_des = 0.5*np.cos(np.pi*t/60)*(np.pi/60)**2
    t2dd_des = -0.5*np.cos(np.pi*t/20)*(np.pi/20)**2
    qddDes = np.array([[t1dd_des], [t2dd_des]])

    qtilda = np.array([[t1-t1_des], [t2-t2_des]])
    qdtilda = np.array([[t1d-t1d_des], [t2d-t2d_des]])

    S = qdtilda + lamda*qtilda
    qdr = qdDes - lamda * qtilda
    qddr = qddDes - lamda*qdtilda
    # the Controlling Variables Part 2
    T = np.matmul(M,qddDes) + np.matmul(C, qdDes)  + G - np.matmul(Kd, qdtilda) - np.matmul(Kp, qtilda)
    dq2dt = np.matmul(np.linalg.inv(M),T-np.matmul(C, dq1dt) - G)

    return dq1dt[0][0], dq1dt[1][0], dq2dt[0][0], dq2dt[1][0]

sol = odeint(equ1, [0.1, 0.1, 0.2, 0.2], t, args=(Kp,Kd))
dq2dt = np.zeros((len(t), 2))
for i in range(len(t)):
    out = equ1(sol[i,:], t[i],Kp,Kd)
    dq2dt[i,0], dq2dt[i,1] = out[2], out[3]


#plots
plt.figure(figsize=(6,7))
plt.subplot(3,1,1)
plt.title('Controlled Torque')
philine, = plt.plot(t, sol[:,0])
thetaline, = plt.plot(t, sol[:,1])
plt.grid()
plt.legend([thetaline, philine],
           ['Theta2', 'Theta1'])
plt.subplot(3,1,2)
t1_des = 0.5*(1-np.cos(np.pi*t/60))
t2_des = 0.5*(1+np.cos(np.pi*t/20))
phidotline, = plt.plot(t, t1_des)
thetadotline, = plt.plot(t, t2_des)
plt.legend([thetadotline, phidotline],
           ['Theta2_Des', 'Theta1_Des'])
plt.grid()
plt.subplot(3,1,3)

philine, = plt.plot(t, sol[:,0]-t1_des)
thetaline, = plt.plot(t, sol[:,1]-t2_des)
plt.grid()
plt.legend([thetaline, philine],
           ['Theta2_error', 'Theta1_Error'])
plt.xlabel('Time (s)')
#plt.savefig('Controlled Torque model.jpg')
plt.show()
