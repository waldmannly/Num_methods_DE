import matplotlib.pyplot as plt
import numpy as np
import math


def Euler_Method( y0,  x0,  h, steps, f ):
    steps = steps -1
    y000=[]
    while ( steps >= 0 ):
        y0 = y0 + h * f(y0)
        x0 = x0+  h
        y000.append(y0)
        steps = steps-1
    return y000

def Improved_Euler_Method( y0,  x0,  h, steps , f) :
    h2 = 0.5 * h
    steps = steps -1
    y000=[]
    while ( steps >= 0 ):
        k1 = h2 * f(y0)
        y0 = y0+  h * f( y0 + k1)
        x0 = x0+  h
        y000.append(y0)
        steps = steps-1
    return y000

def classical_RK(y0,  x0,  h, steps, f):
    h2 = 0.5 * h;
    steps = steps-1
    y000=[]
    while ( steps >= 0 ) :
      k1 = h * f(y0)
      k2 = h * f( y0 + 0.5 * k1)
      k3 = h * f( y0 + 0.5 * k2)
      x0 = x0+ h
      k4 = h * f( y0 + k3)
      y0 = y0 + 1/6 * ( k1 + k2 + k2 + k3 + k3 + k4 )
      y000.append(y0)
      steps = steps-1
    return y000

def getArraysFromList(y,n):
    numArrays = len(y[1])
    l = [None] * numArrays
    for t in np.arange(0,numArrays) :
        u=[i[t] for i in y]
        l[t] = u
    return l

n= 50
x= 0
h= 2*math.pi/n # (2pi -0) /n  length of the interval over the number of steps
step= n

y0 = np.array([1,0])
def f(y):
    return np.array([y[1], -y[0]])

yn1 = Euler_Method(y0,x,h,step, f)
yn1 = getArraysFromList(yn1,n)
print("Euler_Method:")
print(yn1)

yn2 = Improved_Euler_Method(y0,x,h,step,f)
yn2 = getArraysFromList(yn2,n)
print("Improved_Euler_Method:")
print(yn2)

yn3 = classical_RK(y0,x,h,step,f)
yn3 = getArraysFromList(yn3,n)
print("classical_RK:")
print(yn3)

plt.figure()
plt.plot(np.linspace(0, 2*math.pi, n), yn1[0], label= 'Euler')
plt.plot(np.linspace(0, 2*math.pi, n), yn2[0], label= 'Improved_Euler_Method')
plt.plot(np.linspace(0, 2*math.pi, n), yn3[0], label = "RK ")
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(np.linspace(0, 2*math.pi, n), yn1[1], label= 'Euler')
plt.plot(np.linspace(0, 2*math.pi, n), yn2[1], label= 'Improved_Euler_Method')
plt.plot(np.linspace(0, 2*math.pi, n), yn3[1], label = "RK ")
plt.legend(loc='best')
plt.show()
