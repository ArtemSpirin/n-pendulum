import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter
from numpy.linalg import solve


n = 4

m = list()
L = list()
the = list()
the_d = list()
the_dd = list()
x_s = list()
y_s = list()
t, g = smp.symbols('t g')
for i in range(1,n+1):
    L.append(smp.symbols('L'  +'_'+str(i)))
    m.append(smp.symbols('m' +'_'+str(i)))
    the.append(smp.symbols(r'\theta' +'_'+ str(i), cls=smp.Function))
    the[i-1] = the[i-1](t)
    the_d.append(smp.diff(the[i-1], t))
    the_dd.append(smp.diff(the_d[i-1], t))
x_s.append(L[0] * smp.sin(the[0]))
y_s.append(-L[0] * smp.cos(the[0]))
for i in range(1, n):
    x_s.append(L[i] * smp.sin(the[i]))
    x_s[i] += x_s[i - 1]
    y_s.append(-L[i] * smp.cos(the[i]))
    y_s[i] += y_s[i - 1]
T = [1 / 2 * m[i] * (smp.diff(x_s[i], t) ** 2 + smp.diff(y_s[i], t) ** 2) for i in range(n)]
V = [m[i] * g * y_s[i] for i in range(n)]
Lag = sum(T) - sum(V)
LE = [smp.diff(Lag, the[i]) - smp.diff(smp.diff(Lag, the_d[i]), t).simplify() for i in range(n)]

Sols = smp.solve( LE, tuple(the_dd))

dzdt_f = [smp.lambdify((t, g, *m, *L, *the, *the_d), Sols[the_dd[i]],"scipy") for i in range(n)]

dthedt_f = [smp.lambdify(the_d[i], the_d[i],"scipy") for i in range(n)]

def dSdt(S, t, g, *mL):
    the = S[::2]
    z = S[1::2]
    Ans = list()
    for i in range(n):
        Ans.append(dthedt_f[i](z[i]))
        Ans.append(dzdt_f[i](t, g, *mL[:n], *mL[n:], *the, *z))


    return Ans
#{Derivative(\theta_1(t), (t, 2)): -1.0*L1*m2*sin(\theta_1(t) - \theta_2(t))*cos(\theta_1(t) - \theta_2(t))*Derivative(\theta_1(t), t)**2/(1.0*L1*m1 - 1.0*L1*m2*cos(\theta_1(t) - \theta_2(t))**2 + 1.0*L1*m2) + 1
t = np.linspace(0, 40, 1001)
g = 9.81
m = [1 for i in range(n)]  # [i + 1 for i in range(n)]
L = [1 for i in range(n)]  # [(i + 1)/2 for i in range(n)]
y0 = [-np.pi/3, 0, np.pi/3, 0,-np.pi/3, 0,np.pi/3, 0]  # [1 for i in range(2*n)]
ans = odeint(dSdt, y0, t=t, args=(g, *m, *L))


def get_x_and_y_s(t, the, L):
    the = [ans.T[i * 2] for i in range(n)]
    ans1 = list()
    ans1.append(L[0] * np.sin(the[0]))
    ans1.append(-L[0] * np.cos(the[0]))
    for i in range(1, n):
        ans1.append(L[i] * np.sin(the[i]))
        ans1[i*2] += ans1[(i - 1)*2]
        ans1.append(-L[i] * np.cos(the[i]))
        ans1[i*2+1] += ans1[(i - 1)*2+1]
    return tuple(ans1)


x_s = [get_x_and_y_s(t, the, L)[2*i] for i in range(n)]
y_s = [get_x_and_y_s(t, the, L)[2*i+1] for i in range(n)]


def animate(i):
    ln1.set_data([0, *[x[i] for x in x_s]], [0, *[y[i] for y in y_s]])


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_facecolor('w')
ax.get_xaxis().set_ticks([])  # enable this to hide x axis ticks
ax.get_yaxis().set_ticks([])  # enable this to hide y axis ticks
ln1, = plt.plot([], [], 'ko-', lw=3, markersize=8)
ax.set_ylim(-4, 4)
ax.set_xlim(-4, 4)
ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50)

