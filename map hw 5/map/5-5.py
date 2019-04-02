import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import math
from numpy.linalg import *

def Euler_Method( y0, y1,  x0,  h, steps ):
    steps = steps -1
    y000=[]
    y111=[]
    while ( steps >= 0 ):
        y0 = y0 + h * y1
        y1 = y1 + h * -y0
        x0 = x0+  h
        y000.append(y0)
        y111.append(y1)
        steps = steps-1
    print(x0)
    # return (y0,y1)
    return (y000,y111)

def Improved_Euler_Method( y0, y1,  x0,  h, steps ) :
    def f0(y): return y
    def f1(y): return -y
    h2 = 0.5 * h
    steps = steps -1
    y000=[]
    y111=[]
    while ( steps >= 0 ):
        k1 = h2 * f0(y1)
        y0 = y0+  h * f0( y1 + k1)
        kk1= h2 * f1(y0)
        y1 = y1 + h* f1(y0+ kk1)
        x0 = x0+  h
        y000.append(y0)
        y111.append(y1)
        steps = steps-1
    return (y000,y111)

def classical_RK(y0, y1,  x0,  h, steps):
    def f(y): return y
    def f1(y): return -y
    h2 = 0.5 * h;
    steps = steps-1
    y000=[]
    y111=[]
    while ( steps >= 0 ) :
      k1 = h * f(y1)
      k2 = h * f( y1 + 0.5 * k1)
      k3 = h * f( y1 + 0.5 * k2)
      x0 = x0+ h
      k4 = h * f( y1 + k3)
      y0 = y0 + 1/6 * ( k1 + k2 + k2 + k3 + k3 + k4 )

      kk1 = h * f1(y0)
      kk2 = h * f1(y0 + 0.5 * kk1)
      kk3 = h * f1(y0 + 0.5 * kk2)
      kk4 = h * f1(y0 + kk3)
      y1 = y1 + 1/6 * ( kk1 + kk2 + kk2 + kk3 + kk3 + kk4 )

      y000.append(y0)
      y111.append(y1)
      steps = steps-1
    return (y000,y111)

n= 100
y00=1
y11 =0
x= 0
h= 2*math.pi/n
step= n

yn1 = Euler_Method(y00, y11,x,h,step)
print("Euler_Method:")
print(yn1)

yn2 = Improved_Euler_Method(y00, y11,x,h,step)
print("Improved_Euler_Method:")
print(yn2)

yn3 = classical_RK(y00, y11,x,h,step)
print("classical_RK:")
print(yn3)

plt.figure()
plt.plot(np.linspace(0, 2*math.pi, n), yn1[0], label= 'Euler')
plt.plot(np.linspace(0, 2*math.pi, n), yn2[0], label= 'Improved_Euler_Method')
plt.plot(np.linspace(0, 2*math.pi, n), yn3[0], label = "RK ")
plt.show()
