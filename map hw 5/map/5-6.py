import matplotlib.pyplot as plt
import numpy as np
import math

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

# n=50 # number of desired steps
y00=1 # inital y
x= 0 # initial x
# h= geth(0,2,n) # geth(interval end ponts, number of steps)
# step= n
def f(y): return -20*y # given function

step=20
yn20 = Euler_Method(y00,x,geth(0,2,step),step, f)

step=30
yn30 = Euler_Method(y00,x,geth(0,2,step),step, f)

step=50
yn50 = Euler_Method(y00,x,geth(0,2,step),step, f)

step=100
yn100 = Euler_Method(y00,x,geth(0,2,step),step, f)

plt.figure()
plt.plot(np.linspace(0,2,20), yn20, label ="n=20")
plt.plot(np.linspace(0,2,30), yn30, label ="n=30")
plt.plot(np.linspace(0,2,50), yn50, label ="n=50")
plt.plot(np.linspace(0,2,100), yn100, label ="n=100")
plt.legend(loc='best')
plt.show()


print("""We can see that for small n, this method is unstable. I would
      assume that this is caused by a large derivative from our f function. Once
      we increase n to make h small enough, we get a reasonable result. """)
