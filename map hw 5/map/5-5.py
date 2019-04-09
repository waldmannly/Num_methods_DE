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

# these will contain the values for part 2
sEM =[]
sIEM =[]
sRK =[]

# part 1
for n in np.arange(10,110, 10) :
    x= 0
    h= 2*math.pi/n # (2pi -0) /n  length of the interval over the number of steps
    step= n

    y0 = np.array([1,0])
    def f(y):
        return np.array([y[1], -y[0]])

    yn1 = Euler_Method(y0,x,h,step, f)
    yn1 = getArraysFromList(yn1,n)
    sEM.append(yn1[0][n-1]) # steal the last value for each of the methods 

    yn2 = Improved_Euler_Method(y0,x,h,step,f)
    yn2 = getArraysFromList(yn2,n)
    sIEM.append(yn2[0][n-1])

    yn3 = classical_RK(y0,x,h,step,f)
    yn3 = getArraysFromList(yn3,n)
    sRK.append(yn3[0][n-1])

# n= 100 is the last number of steps.
plt.figure()
plt.plot(np.linspace(0, 2*math.pi, n), yn1[0], label= 'y0 - Euler')
plt.plot(np.linspace(0, 2*math.pi, n), yn2[0], label= 'y0 - Improved_Euler_Method',linewidth=4)
plt.plot(np.linspace(0, 2*math.pi, n), yn3[0], label = "y0 - RK ",linewidth=2)
plt.legend(loc='best')
# plt.savefig("5-5-y0-solution-out.png")
plt.show()

plt.figure()
plt.plot(np.linspace(0, 2*math.pi, n), yn1[1], label= 'y1 - Euler')
plt.plot(np.linspace(0, 2*math.pi, n), yn2[1], label= 'y1 - Improved_Euler_Method',linewidth=4)
plt.plot(np.linspace(0, 2*math.pi, n), yn3[1], label = "y1 - RK ",linewidth=2)
plt.legend(loc='best')
# plt.savefig("5-5-y1-solution-out.png")
plt.show()

#part 2 error plot
plt.figure()
plt.loglog(np.arange(10,110, 10), np.abs(sEM-np.cos(math.pi*2)), label="Euler at 2pi" )
plt.loglog(np.arange(10,110, 10), np.abs(sIEM-np.cos(math.pi*2)), label="Improved Euler at 2pi" )
plt.loglog(np.arange(10,110, 10), np.abs(sRK-np.cos(math.pi*2)), label="RK at 2pi" )
plt.legend(loc='best')
# plt.savefig("5-5-y0-error-out.png")
plt.show()
