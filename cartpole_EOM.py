import numpy as np
import matplotlib.pyplot as plt

m1=1
m2=1
l=0.5
g=9.8
J=0.5
F=0
tau=0.01

def f(z,m1,m2,l,J,F,g):
    
    v,omega,x,theta=z

    A1=np.array([[m1+m2,m2*l*np.cos(theta)],[m2*l*np.cos(theta),m2*l**2+J]])
    A2=np.array([[m2*l*omega**2*np.sin(theta)+F],[m2*g*l*np.sin(theta)]])
    A1_inv=np.linalg.inv(A1)

    h=A1_inv@A2

    return np.array([h[0,0],h[1,0],v,omega])


def RungeKutta(z,tau,m1,m2,l,J,F,g):
    k1=f(z,m1,m2,l,J,F,g)
    k2=f(z+k1*tau/2,m1,m2,l,J,F,g)
    k3=f(z+k2*tau/2,m1,m2,l,J,F,g)
    k4=f(z+k3*tau,m1,m2,l,J,F,g)
    return z+tau/6*(k1+2*k2+2*k3+k4)


z0=np.array([0.0,0.0,0.0,0.1])
T=20.0
N=int(T/tau)
z_list=[z0]

for i in range(N):
    z=RungeKutta(z_list[i],tau,m1,m2,l,J,F,g)
    
    z_list.append(z)
    
time=np.arange(N+1)*tau
z_array=np.array(z_list)

plt.plot(time,z_array[:,2],label="Position")
plt.plot(time,z_array[:,3],label="Theta")
plt.xlabel("Time [s]")
plt.ylabel("value")
plt.legend()
plt.show()

