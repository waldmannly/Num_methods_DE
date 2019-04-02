import matplotlib.pyplot as plt
import numpy as np

def Euler_Method( y0,  x0,  h, steps ):
    def f(y): return -20*y
    steps = steps -1
    y000=[]
    while ( steps >= 0 ):
        y0 = y0 + h * f(y0)
        x0 = x0+  h
        y000.append(y0)
        steps = steps-1
    return (y000)

def geth(t0,t1, num):
    range = t1-t0
    points = range/num
    return (points)

n=50

y00=1
x= 0
h= geth(0,2,n)
step= n

yn2 = Euler_Method(y00,x,h,step)
print("Euler_Method:")
print(yn2)

plt.figure()
plt.plot(np.linspace(0,2,n), yn2)
plt.show()
