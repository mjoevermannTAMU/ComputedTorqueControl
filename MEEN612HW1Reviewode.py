import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

a = 10
b = 3
g = 2
d = 5
t = np.linspace(0, 3, 100)
Kp = 1
Kd = 10

pi= np.pi
X = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
             [7*0**6, 6*0**5, 5*0**4, 4*0**3, 3*0**2, 2*0**1, 1, 0],
             [1**7, 1**6, 1**5, 1**4, 1**3, 1**2, 1**1, 1**0],
             [7*1**6, 6*1**5, 5*1**4, 4*1**3, 3*1**2, 2*1**1, 1**0, 0],
             [2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2**1, 2**0],
             [7*2**6, 6*2**5, 5*2**4, 4*2**3, 3*2**2, 2*2**1, 2**0, 0],
             [3**7, 3**6, 3**5, 3**4, 3**3, 3**2, 3**1, 3**0],
             [7*3**6, 6*3**5, 5*3**4, 4*3**3, 3*3**2, 2*3**1, 3**0, 0]])

phi0 = np.array([[0],
              [0],
              [pi/3],
              [1],
              [2*pi/3],
              [2],
              [pi],
              [0]])
theta0 = np.array([[0],
              [0],
              [pi/2],
              [1],
              [3*pi/2],
              [1],
              [2*pi],
              [0]])

phiCoeff = np.matmul(np.linalg.inv(X), phi0)
thetaCoeff = np.matmul(np.linalg.inv(X), theta0)

def pathpoly(t, coeff):
    poly = 0.0
    n = len(coeff)
    for i in range(n):
        if n < 0:
            break
        elif t == 0.0:
            poly+=poly
        else:
            poly += coeff[i]*t**(n-i)
    return float(poly)
def pathpolyDerivative(t, coeff):
    poly = 0.0
    n = len(coeff)
    for i in range(n):
        if n < 0:
            break
        elif t == 0.0:
            poly+=poly
        else:
            poly += (n-i)*coeff[i] * t ** (n-i-1)
    return float(poly)
def pathpolyDDerivative(t, coeff):
    poly = 0.0
    n = len(coeff)
    for i in range(n):
        if n < 0:
            break
        elif t == 0.0:
            poly+=poly
        else:
            poly += (n-i-1)*(n-i)* coeff[i] * t ** (n-i-2)
    return float(poly)

def equ1(q, t, Kp,Kd):
    phi, theta, phidot, thetadot = q
    # the controlling variables Part 1
    #T = -Kp*np.array([[phi], [theta]])-Kd*np.array([[phidot], [thetadot]])

    # the first equation
    dq1dt = np.array([[phidot], [thetadot]])
    q1 = np.array([[phi], [theta]])
    # the second equation
    M = np.array([[a+b*np.sin(theta)**2, g*np.cos(theta)], [g*np.cos(theta), b]])
    C = np.array([[b*thetadot*np.sin(theta)*np.cos(theta), -g*thetadot*np.sin(theta)+b*phidot*np.sin(theta)*np.cos(theta)],
                [-b*phidot*np.sin(theta)*np.cos(theta), 0]])
    G = np.array([[0], [-d*np.sin(theta)]])
    # the Controlling Variables Part 2
    qdddot = np.array([[pathpolyDDerivative(t, phiCoeff)],
                       [pathpolyDDerivative(t, thetaCoeff)]])
    qddot = np.array([[pathpolyDerivative(t, phiCoeff)],
                      [pathpolyDerivative(t, thetaCoeff)]])
    qd = np.array([[pathpoly(t, phiCoeff)],
                   [pathpoly(t, thetaCoeff)]])

    T = np.matmul(M, qdddot)+np.matmul(C, qddot)-Kd*(dq1dt-qddot)-Kp*(q1 - qd)+ G

    dq2dt = np.matmul(np.linalg.inv(M),(T-np.matmul(C, dq1dt) - G))
    return dq1dt[0][0], dq1dt[1][0], dq2dt[0][0], dq2dt[1][0]

sol = odeint(equ1, [1, -1, 0, 0], t, args=(Kp,Kd))
dq2dt = np.zeros((len(t), 2))
for i in range(len(t)):
    out = equ1(sol[i,:], t[i],Kp,Kd)
    dq2dt[i,0], dq2dt[i,1] = out[2], out[3]


#plots
plt.figure(figsize=(6,7))
plt.subplot(3,1,1)
plt.title('Path Controlled; Kp={} Kd={}'.format(Kp,Kd))
philine, = plt.plot(t, sol[:,0])
thetaline, = plt.plot(t, sol[:,1])
plt.grid()
plt.legend([thetaline, philine],
           ['Theta', 'Phi'])
plt.subplot(3,1,2)
phidotline, = plt.plot(t, sol[:,2])
thetadotline, = plt.plot(t, sol[:,3])
plt.legend([thetadotline, phidotline],
           ['ThetaDot', 'PhiDot'])
plt.grid()
plt.subplot(3,1,3)
philine, = plt.plot(t, dq2dt[:,0])
thetaline, = plt.plot(t, dq2dt[:,1])
plt.grid()
plt.legend([thetaline, philine],
           ['ThetaDoubleDot', 'PhiDoubleDot'])
plt.savefig('Path controlled model.jpg')
plt.show()
