import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


m1 = 5
m2Guess = 10
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

gamma1, gamma2 = 2,4

def equ1(q, t):
    t1, t2, t1d, t2d, m2 = q
    # the controlling variables Part 1
    # Desired Path variables
    t1_des = 0.5 * (1 - np.cos(np.pi * t / 60))
    t2_des = 0.5 * (1 + np.cos(np.pi * t / 20))
    qDes = np.array([[t1_des], [t2_des]])
    t1d_des = 0.5 * np.sin(np.pi * t / 60) * np.pi / 60
    t2d_des = -0.5 * np.sin(np.pi * t / 20) * np.pi / 20
    qdDes = np.array([[t1d_des], [t2d_des]])
    t1dd_des = 0.5 * np.cos(np.pi * t / 60) * (np.pi / 60) ** 2
    t2dd_des = -0.5 * np.cos(np.pi * t / 20) * (np.pi / 20) ** 2
    qddDes = np.array([[t1dd_des], [t2dd_des]])

    qtilda = np.array([[t1 - t1_des], [t2 - t2_des]])
    qdtilda = np.array([[t1d - t1d_des], [t2d - t2d_des]])

    S = qdtilda + lamda * qtilda
    qdr = qdDes - lamda * qtilda
    qddr = qddDes - lamda * qdtilda

    # the first equation
    qd = np.array([[t1d], [t2d]])
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

    u = np.matmul(M, qddr)+ np.matmul(C, qdr) + G - np.matmul(Kd, S)

    # the regressor vectors
    y01 = (m1 * l1 ** 2 + Iyy2 * np.cos(t2) ** 2 + Iyy1 + Ixx2 * np.sin(t2) ** 2) * qddr[0][0] + 2 * (Ixx2 - Iyy2) * np.sin(
        t2) * np.cos(t2) * qdr[0][0] * qdr[1][0]
    y02 = Izz * qddr[1][0] + (Iyy2 - Ixx2) * np.sin(t2) * np.cos(t2) * qdr[0][0] ** 2
    Y0 = np.array([[y01], [y02]])

    y11 = (L1 + l2 * np.cos(t2)) ** 2 * qddr[0][0] - 2 * (L1 + l2 * np.cos(t2)) * l2 * np.sin(t2) * qdr[1][0] * qdr[0][0]
    y12 = l2 ** 2 * qdr[1][0] + (L1 + l2 * np.cos(t2)) * l2 * np.sin(t2) * qdr[0][0] ** 2 + g * L2 * np.cos(t2)
    Y1 = np.array([[y11], [y12]])

    mhatd = -np.matmul(np.transpose(S), Y1) / gamma2

    # the Controlling Variables Part 2
    dq2dt = np.matmul(np.linalg.inv(M),u-np.matmul(C, qd) - G)


    return qd[0][0], qd[1][0], dq2dt[0][0], dq2dt[1][0], mhatd

sol = odeint(equ1, [0.1, 0.1, 0.2, 0.2, 0.5*m2Guess], t)
dq2dt = np.zeros((len(t), 2))
for i in range(len(t)):
    out = equ1(sol[i,:], t[i])
    dq2dt[i,0], dq2dt[i,1] = out[2], out[3]


#plots
plt.figure(figsize=(6,7))
plt.subplot(4,1,1)
plt.title('Adaptive Parameter Control')
philine, = plt.plot(t, sol[:,0])
thetaline, = plt.plot(t, sol[:,1])
plt.grid()
plt.legend([thetaline, philine],
           ['Theta2', 'Theta1'])
plt.subplot(4,1,2)
t1_des = 0.5*(1-np.cos(np.pi*t/60))
t2_des = 0.5*(1+np.cos(np.pi*t/20))
phidotline, = plt.plot(t, t1_des)
thetadotline, = plt.plot(t, t2_des)
plt.legend([thetadotline, phidotline],
           ['Theta2_Des', 'Theta1_Des'])
plt.grid()
plt.subplot(4,1,3)

philine, = plt.plot(t, sol[:,0]-t1_des)
thetaline, = plt.plot(t, sol[:,1]-t2_des)
plt.grid()
plt.legend([thetaline, philine],
           ['Theta2_error', 'Theta1_Error'])
plt.xlabel('Time (s)')
plt.subplot(4, 1,4)
massGuesses, = plt.plot(t, sol[:,4]-m2Guess)
plt.legend([massGuesses],['Mass Error'])
plt.grid()
plt.savefig('mass adaptive control model.jpg')
plt.show()
