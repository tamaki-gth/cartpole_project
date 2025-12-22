import numpy as np

m1=1
m2=1
l=0.5
g=9.8
J=0.5
F=0
tau=0.1

def f(z,m1,m2,l,J,F,g):
    
    x,x_dot,theta,theta_dot=z

    A1=np.array([[m1+m2,m2*l*np.cos(theta)],[m2*l*np.cos(theta),m2*l**2+J]])
    A2=np.array([[m2*l*theta_dot**2*np.sin(theta)+F],[m2*g*l*np.sin(theta)]])
    A1_inv=np.linalg.inv(A1)

    h=A1_inv@A2

    return np.array([h[0,0],h[1,0],x_dot,theta_dot])


def RungeKutta(z,tau,m1,m2,l,J,F,g):
    k1=f(z,m1,m2,l,J,F,g)
    k2=f(z+k1*tau/2,m1,m2,l,J,F,g)
    k3=f(z+k2*tau/2,m1,m2,l,J,F,g)
    k4=f(z+k3*tau,m1,m2,l,J,F,g)
    return z+tau/6*(k1+2*k2+2*k3+k4)

