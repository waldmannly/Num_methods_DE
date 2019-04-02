
def Improved_Euler_Method( y0,  x0,  h, steps ) :
    def f(x,y): return 2*(y*y +1)/(x*x +4)
    h2 = 0.5 * h
    steps = steps -1
    while ( steps >= 0 ):
        k1 = h2 * f(x0,y0)
        y0 = y0+  h * f(x0+h2, y0 + k1)
        x0 = x0+  h
        steps = steps-1
    return y0

def Heun_Method( y0,  x0,  h, steps ):
    def f(x,y): return 2*(y*y +1)/(x*x +4)
    steps = steps -1
    while ( steps >= 0 ):
        k1 = h * f(x0,y0)
        y0 = y0+  .5 *(k1 + h * f(x0+h, y0 + k1))
        x0 = x0+  h
        steps = steps-1
    return y0

def Euler_Method( y0,  x0,  h, steps ):
    def f(x,y): return 2*(y*y +1)/(x*x +4)
    steps = steps -1
    while ( steps >= 0 ):
        y0 = y0 + h * f(x0, y0)
        x0 = x0+  h
        steps = steps-1
    return y0

def classical_RK(y0,  x0,  h, steps):
    def f(x,y): return 2*(y*y +1)/(x*x +4)
    h2 = 0.5 * h;
    steps = steps-1
    while ( steps >= 0 ) :
      k1 = h * f(x0,y0)
      k2 = h * f(x0+h2, y0 + 0.5 * k1)
      k3 = h * f(x0+h2, y0 + 0.5 * k2)
      x0 = x0+ h
      k4 = h * f(x0, y0 + k3)
      y0 = y0 + 1/6 * ( k1 + k2 + k2 + k3 + k3 + k4 )
      steps = steps-1
    return y0

y=1
x= 0
h=.1
step= 10

yn = Improved_Euler_Method(y,x,h,step)
print("Improved_Euler_Method:")
print(yn)

yn1 = Heun_Method(y,x,h,step)
print("Heun_Method:")
print(yn1)

yn2 = Euler_Method(y,x,h,step)
print("Euler_Method:")
print(yn2)

yn3 = classical_RK(y,x,h,step)
print("classical_RK:")
print(yn3)
