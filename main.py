from sympy import symbols
import sympy as smp
from sympy.physics.mechanics import *
from sympy import Dummy, lambdify
from numpy import array, hstack, zeros, linspace, pi, ones
from numpy.linalg import solve
from scipy.integrate import odeint
from numpy import zeros, cos, sin, arange, around
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
import math


n = 7

q = dynamicsymbols('q:' + str(n + 1))
qd = dynamicsymbols('q:' + str(n + 1),1)
u = dynamicsymbols('u:' + str(n + 1))
f = dynamicsymbols('f')

m = symbols('m:' + str(n + 1))
l = symbols('l:' + str(n))
g, t = symbols('g t')








I = ReferenceFrame('I')
O = Point('O')
O.set_vel(I, smp.sin(t)*I.x)
P0 = Point('P0')

# Hinge point of top link
P0.set_pos(O,  ( q[0]) * I.x)
P0.set_vel(I, qd[0] * I.x)
Pa0 = Particle('Pa0', P0, m[0])
frames = [I]
points = [P0]
particles = [Pa0]
forces = [(P0, f*I.x- m[0] * g * I.y)]


for i in range(n):
    Bi = I.orientnew('B' + str(i), 'Axis', [q[i + 1], I.z])
    Bi.set_ang_vel(I, qd[i + 1] * I.z)
    frames.append(Bi)

    Pi = points[-1].locatenew('P' + str(i + 1), l[i] * Bi.x)
    Pi.v2pt_theory(points[-1], I, Bi)
    points.append(Pi)

    Pai = Particle('Pa' + str(i + 1), Pi, m[i + 1])
    particles.append(Pai)
    forces.append((Pi, -m[i + 1] * g * I.y))


Lag = Lagrangian(I, *particles)

LM = LagrangesMethod(Lag, q, forces, frames)
LM.form_lagranges_equations()

arm_length = 1. / n
bob_mass = 0.01 / n
parameters = [g, m[0]]
parameter_vals = [9.81, 100 / n]
for i in range(n):
    parameters += [l[i], m[i + 1]]
    parameter_vals += [arm_length, bob_mass]
dynamic = q + qd
dynamic.append(f)


dummy_symbols = [Dummy() for i in dynamic]
dummy_dict = dict(zip(dynamic, dummy_symbols))
kindiff_dict = kane.kindiffdict()
M = LM.mass_matrix_full.subs(Lag).subs(dummy_dict)
F = LM.forcing_full.subs(Lag).subs(dummy_dict)

M_func = lambdify(dummy_symbols + parameters, M)
F_func = lambdify(dummy_symbols + parameters, F)


def right_hand_side(x, t, args):
    f =  smp.sin(t)
    u = 0.0
    arguments = hstack((x, u, args))
    dx = array(solve(M_func(*arguments),
                     F_func(*arguments))).T[0]

    return dx

x0 = hstack(( 0, pi / 4 * ones(len(q) - 1), 1e-3 * ones(len(qd)) )) # Initial conditions, q and u
t = linspace(0, 10, 1000)                                          # Time vector
y = odeint(right_hand_side, x0, t, args=(parameter_vals,))

def animate_pendulum(t, states, length, filename='pen.gif'):


    numpoints = states.shape[1] / 2

    fig = plt.figure()


    cart_width = 0.4
    cart_height = 0.2


    xmin = around(states[:, 0].min() - cart_width / 2.0, 1)-3
    xmax = around(states[:, 0].max() + cart_width / 2.0, 1)+3


    ax = plt.axes(xlim=(xmin, xmax), ylim=(-1.1, 1.1), aspect='equal')


    time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)


    rect = Rectangle([states[0, 0] - cart_width / 2.0, -cart_height / 2],
                     cart_width, cart_height, fill=True, color='red', ec='black')
    ax.add_patch(rect)


    line, = ax.plot([], [], lw=2, marker='o', markersize=6)

    def init():
        time_text.set_text('')
        rect.set_xy((0.0, 0.0))
        line.set_data([], [])
        return time_text, rect, line,

    def animate(i):
        time_text.set_text('time = {:2.2f}'.format(t[i]))
        rect.set_xy((states[i, 0]  - cart_width / 2.0, -cart_height / 2))  # Add amplitude to set the new position
        x = hstack((states[i, 0] , zeros(int(numpoints - 1))))  # Add amplitude to x to match the new position
        y = zeros(int(numpoints))
        for j in arange(1, numpoints):
            x[int(j)] = x[int(j - 1)] + length * cos(states[int(i), int(j)])
            y[int(j)] = y[int(j - 1)] + length * sin(states[int(i), int(j)])
        line.set_data(x, y)
        return time_text, rect, line,

    def animate(i):
        time_text.set_text('time = {:2.2f}'.format(t[i]))
        rect.set_xy((states[i, 0] - cart_width / 2.0, -cart_height / 2))
        x = hstack((states[i, 0], zeros(int(numpoints - 1))))
        y = zeros(int(numpoints))
        for j in arange(1, numpoints):
            x[int(j)] = x[int(j - 1)] + length * cos(states[int(i), int(j)])
            y[int(j)] = y[int(j - 1)] + length * sin(states[int(i), int(j)])
        line.set_data(x, y)
        return time_text, rect, line,

    anim = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
                                   interval=t[-1] / len(t) * 1000, blit=True, repeat=True)


    if filename is not None:
        ani.save('pen.gif', writer='pillow', fps=25)


animate_pendulum(t, y, arm_length, filename="pen.gif")
