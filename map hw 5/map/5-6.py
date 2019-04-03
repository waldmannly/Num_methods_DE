import matplotlib.pyplot as plt
import numpy as np

def Euler_Method( y0,  x0,  h, steps,f ):
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

n=50 # number of desired steps
y00=1 # inital y
x= 0 # initial x
h= geth(0,2,n) # geth(interval end ponts, number of steps)
step= n
def f(y): return -20*y # given function

yn2 = Euler_Method(y00,x,h,step, f)
print("Euler_Method:")
print(yn2)

plt.figure()
plt.plot(np.linspace(0,2,n), yn2)
plt.show()
