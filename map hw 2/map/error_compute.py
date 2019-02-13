from math import sin, cos, exp
import matplotlib.pyplot as plt
import numpy as np

def Dplus(f, x = 1):
    def g(h):
        return (f(x+h)-f(x))/h
    return g

def Dminus(f, x=1):
    def p(h):
        return (f(x)-f(x-h))/h
    return p

def Dcenter(f, x = 1):
    def k(h):
        return (f(x+h)-f(x-h))/(2*h)
    return k


g = Dplus(sin)
p = Dminus(sin)
k= Dcenter(sin)

print("calculating values: ")
print("\nD+: ")
print(g(.00001))
print("\nD-: ")
print(p(.00001))
print("\nD0: ")
print(k(.00001))
print("\nexact: ")
print(cos(1))

# part 1
x = np.linspace(1,40,40)

f = np.cos(np.ones(40))

p1= [p(exp(-1*i)) for i in x]
g1= [g( exp(-1*i) ) for i in x]
k1= [k(exp(-1*i)) for i in x]
x= np.exp(-1*x)

print("exp(-n):")

#part 2
print("\nPart 2 - (np.finfo(abs(g1-f).dtype).eps):")
print("|D+ - cos(x)| : ")
print(np.finfo(abs(g1-f).dtype).eps)
print("\n|D- - cos(x)| : ")
print(np.finfo(abs(p1-f).dtype).eps)
print("\n|D0 - cos(x)| : ")
print(np.finfo(abs(k1-f).dtype).eps)

print("\nComparing this to the numbers calculated in part a, \
      we can see that the error present in our calculations \
      from using numpy's data type start to out weigh the error\
      from our calcultaions. Thus going to the precision of the \
      the final exp(-35) to exp(-40) is irrelevant because the increase\
      in accuracy is orders of magnitude smaller than the error from \
      the data type we store our answers in. ")


plt.figure()
plt.loglog(x, abs(p1-f), 'b', lw=2, label='D-')
plt.loglog(x, abs(g1-f), 'r', lw=2, label='D+')
plt.loglog(x, abs(k1-f), 'g', lw=2, label='D0')
plt.legend(loc='best')
plt.show()
