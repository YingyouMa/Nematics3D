import numpy as np
import matplotlib.pyplot as plt
import time

from elastic_test import get_deform_n, get_deform_Q, get_deform_Q_divide

# import sys
# sys.path
# sys.path.append(r'E:\Program\GitHub\3D-active-nematics\simulation')
# sys.path

# from Nematics3D.elastic import get_deform_n, get_deform_Q, get_deform_Q_divide

def Plus(*args):
    result = 0
    for i in args:
        result = result + i
    return result  

def Times(*args):
    result = 1
    for i in args:
        result = result * i
    return result  

def Rational(a,b):
    return a/b

def List(*args):
    return list(args)

Sin = np.sin
Cos = np.cos
Power = np.power


N = 200
L = 2

x = np.linspace(-L,L,N)
y = np.linspace(-L,L,N)
z = np.linspace(-L,L,N)

X, Y, Z = np.meshgrid(x,y,z, indexing='ij')

theta = 6*X + 3*Y**2 + Z**3
phi   = X + Y + Z

n = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
n = n.transpose((1,2,3,0))
'''
splay_linear = Plus(Times(6, Y, Cos(theta), Sin(phi)), Times(-1, Plus(Times(3, \
Power(Z, 2)), Sin(phi)), Sin(theta)), Times(Cos(phi), Plus(Times(6, \
Cos(theta)), Sin(theta))))
                                                             
twist_linear = Times(Rational(1, 2), Plus(-1, Power(Cos(theta), 2), Times(12, \
Sin(phi)), Times(-1, Power(Sin(theta), 2)), Times(Sin(phi), \
Sin(Times(2, theta))), Times(Cos(phi), Plus(Times(-12, Y), \
Sin(Times(2, theta))))))

                                         
bend_vector = List(Plus(Times(Sin(theta), Plus(Times(Cos(theta), Plus(-6, Sin(phi), \
Times(6, Power(Sin(phi), 2)))), Times(Power(Sin(phi), 2), \
Sin(theta)))), Times(Cos(phi), Plus(Times(-3, Power(Z, 2), \
Power(Cos(theta), 2)), Times(-6, Y, Cos(theta), Sin(phi), \
Sin(theta)), Times(Sin(phi), Power(Sin(theta), 2))))), Plus(Times(-3, \
Power(Z, 2), Power(Cos(theta), 2), Sin(phi)), Times(-1, Cos(theta), \
Plus(Cos(phi), Times(6, Cos(phi), Sin(phi)), Times(6, Y, \
Power(Sin(phi), 2))), Sin(theta)), Times(-1, Cos(phi), Plus(Cos(phi), \
Sin(phi)), Power(Sin(theta), 2))), Times(3, Sin(theta), \
Plus(Times(Power(Z, 2), Cos(theta)), Times(2, Plus(Cos(phi), Times(Y, \
Sin(phi))), Sin(theta))))) 

bend_vector = np.array(bend_vector).transpose((1,2,3,0))
'''

splay = Power(Plus(Times(6, Y, Cos(theta), Sin(phi)), Times(-1, Plus(Times(3, \
Power(Z, 2)), Sin(phi)), Sin(theta)), Times(Cos(phi), Plus(Times(6, \
Cos(theta)), Sin(theta)))), 2)
splay = splay[1:-1,1:-1,1:-1]
                                                                 
twist = Times(Rational(1, 4), Power(Plus(-1, Power(Cos(theta), 2), Times(12, \
Sin(phi)), Times(-1, Power(Sin(theta), 2)), Times(Sin(phi), \
Sin(Times(2, theta))), Times(Cos(phi), Plus(Times(-12, Y), \
Sin(Times(2, theta))))), 2))
twist = twist[1:-1,1:-1,1:-1]

bend = Plus(Times(9, Power(Sin(theta), 2), Power(Plus(Times(Power(Z, 2), \
Cos(theta)), Times(2, Plus(Cos(phi), Times(Y, Sin(phi))), \
Sin(theta))), 2)), Power(Plus(Times(3, Power(Z, 2), Power(Cos(theta), \
2), Sin(phi)), Times(Cos(theta), Plus(Cos(phi), Times(6, Cos(phi), \
Sin(phi)), Times(6, Y, Power(Sin(phi), 2))), Sin(theta)), \
Times(Cos(phi), Plus(Cos(phi), Sin(phi)), Power(Sin(theta), 2))), 2), \
Power(Plus(Times(Sin(theta), Plus(Times(Cos(theta), Plus(-6, \
Sin(phi), Times(6, Power(Sin(phi), 2)))), Times(Power(Sin(phi), 2), \
Sin(theta)))), Times(Cos(phi), Plus(Times(-3, Power(Z, 2), \
Power(Cos(theta), 2)), Times(-6, Y, Cos(theta), Sin(phi), \
Sin(theta)), Times(Sin(phi), Power(Sin(theta), 2))))), 2))
bend = bend[1:-1,1:-1,1:-1]                   
              
deform_theory = np.array([splay, twist, bend])

def check(n, width, category, if_print=True, divn=3):
    if category == 'n':
        deform, diff = get_deform_n(n, width, if_print=if_print)
    elif category == 'Q':
        deform = get_deform_Q(n, width, 2)
    elif category == 'Q2':
        deform = get_deform_Q_divide(n, width, divn=divn)

    return deform[:3]

start = time.time()
print('analyzing')
deform = check(n, 2*L, 'Q')
print(time.time()-start)


index = np.arange((N-2)**3)
np.random.shuffle(index)
sample = index[:1000]

def check_plot(n):
    plt.figure()
    plt.plot(
            (deform_theory[n].reshape(-1))[sample],
            (deform[n].reshape(-1))[sample],
            'o'
            )
    axis = [0, np.max((deform_theory[n].reshape(-1))[sample])]
    plt.plot(axis, axis)
    
check_plot(0)
check_plot(1)
check_plot(2)









