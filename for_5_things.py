from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy import array, hstack, linspace, pi, ones
from numpy import zeros, cos, sin, arange, around
from numpy.linalg import solve
from scipy.linalg import sqrtm
from scipy.integrate import odeint
from sympy import Dummy, lambdify
from sympy import symbols, det
from sympy.physics.mechanics import *
from sympy.matrices import Matrix
import sympy as smp
import numpy as np
from sympy.plotting import plot_parametric
from sympy.plotting import plot3d_parametric_line
import math

n = 3

q = dynamicsymbols('q:' + str(n + 1))
u = dynamicsymbols('u:' + str(n + 1))
f = dynamicsymbols('f')

m = symbols('m:' + str(n + 1))
l = symbols('l:' + str(n))
g, t = symbols('g t')



I = ReferenceFrame('I')
O = Point('O')
O.set_vel(I, 0)

P0 = Point('P0')
P0.set_pos(O, q[0] * I.x)
P0.set_vel(I, u[0] * I.x)
Pa0 = Particle('Pa0', P0, m[0])


frames = [I]
points = [P0]
particles = [Pa0]
forces = [(P0, f * I.x - m[0] * g * I.y)]
kindiffs = [q[0].diff(t) - u[0]]

for i in range(n):
    Bi = I.orientnew('B' + str(i), 'Axis', [q[i + 1], I.z])
    Bi.set_ang_vel(I, u[i + 1] * I.z)
    frames.append(Bi)

    Pi = points[-1].locatenew('P' + str(i + 1), l[i] * Bi.x)
    Pi.v2pt_theory(points[-1], I, Bi)
    points.append(Pi)

    Pai = Particle('Pa' + str(i + 1), Pi, m[i + 1])
    particles.append(Pai)

    forces.append((Pi, -m[i + 1] * g * I.y))

    kindiffs.append(q[i + 1].diff(t) - u[i + 1])


kane = KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs)
fr, frstar = kane.kanes_equations(particles, forces)

arm_length = 1. / n
bob_mass = 0.0001 / n
parameters = [g, m[0]]
parameter_vals = [9.81, 5 / n]
for i in range(n):
    parameters += [l[i], m[i + 1]]
    parameter_vals += [arm_length, bob_mass]


C_matrix = [[0 for i in range(n+1)] for j in range(n+1)]
for i in range(n+1):
    for j in range(n+1):
        if j==i:
            if i == 0:
                C_matrix[i][j] = 0
            else:
                m_sum = 0
                for z in range(i,n+1):
                    if z == 0:
                        m_sum += parameter_vals[1]
                    else:
                        m_sum += parameter_vals[3]
                C_matrix[i][j] = parameter_vals[0]*parameter_vals[2]*m_sum
        else:
            C_matrix[i][j] = 0
A_matrix = [[0 for i in range(n+1)] for j in range(n+1)]
for i in range(n+1):
    for j in range(n+1):
        #if i==j==0:
            #A_matrix[i][j]+=parameter_vals[1]+parameter_vals[3]*n
        #if i == 0 and j!=0:
            #A_matrix[i][j]+=parameter_vals[2]*parameter_vals[3]*(n-j+1)
        #if j == 0 and i != 0:
            #A_matrix[i][j] += parameter_vals[2] * parameter_vals[3] * (n - i+1)
        if j!=0 and i!=0:
            A_matrix[i][j]+=parameter_vals[2]**2*parameter_vals[3]*(n-i+1)

A = Matrix(A_matrix)
C = Matrix(C_matrix)
w = symbols('w')
M = -w**2*A+C
Eq = M.det()

W = smp.solve(Eq,w)

T = Matrix()
for i in range(len(W)):
    print(1)
    M_0 = M.subs(w,W[i])
    a = [[0],[0],[0],[0]]
    v = solve(M_0,a)
    print(v)




dynamic = q + u
dynamic.append(f)
dummy_symbols = [Dummy() for i in dynamic]
dummy_dict = dict(zip(dynamic, dummy_symbols))
kindiff_dict = kane.kindiffdict()
M = kane.mass_matrix_full.subs(kindiff_dict).subs(dummy_dict)
F = kane.forcing_full.subs(kindiff_dict).subs(dummy_dict)
M_func = lambdify(dummy_symbols + parameters, M)
F_func = lambdify(dummy_symbols + parameters, F)


def right_hand_side(x, t, args):
    """Returns the derivatives of the states.

    Parameters
    ----------
    x : ndarray, shape(2 * (n + 1))
        The current state vector.
    t : float
        The current time.
    args : ndarray
        The constants.

    Returns
    -------
    dx : ndarray, shape(2 * (n + 1))
        The derivative of the state.

    """
    u = 0
    arguments = hstack((x, u, args))
    dx = array(solve(M_func(*arguments),
                     F_func(*arguments))).T[0]

    return dx

x0 = hstack(( 0, -3*pi / 8 * ones(len(q) - 1), 1e-3 * ones(len(u)) ))
t = linspace(0, 15, 1000)
y = odeint(right_hand_side, x0, t, args=(parameter_vals,))
y_s   =  list()
for i in range(len(y)):
    y_s.append(y[i][0])

fig, ax = plt.subplots()
ax.plot(t, y_s)
#plt.show()

def animate_pendulum(t, states, length, filename=None):
    """Animates the n-pendulum and optionally saves it to file.

    Parameters
    ----------
    t : ndarray, shape(m)
        Time array.
    states: ndarray, shape(m,p)
        State time history.
    length: float
        The length of the pendulum links.
    filename: string or None, optional
        If true a movie file will be saved of the animation. This may take some time.

    Returns
    -------
    fig : matplotlib.Figure
        The figure.
    anim : matplotlib.FuncAnimation
        The animation.

    """

    numpoints = states.shape[1] / 2


    fig = plt.figure()


    cart_width = 0.4
    cart_height = 0.2


    xmin = around(states[:, 0].min() - cart_width / 2.0, 1)-1
    xmax = around(states[:, 0].max() + cart_width / 2.0, 1)+1


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
        rect.set_xy((states[int(i), 0] - cart_width / 2.0, -cart_height / 2))
        x = hstack((states[int(i), 0], zeros((int(numpoints) - 1))))
        y = zeros((int(numpoints)))
        for j in arange(1, numpoints):
            x[int(j)] = x[int(j) - 1] + length * cos(states[int(i), int(j)])
            y[int(j)] = y[int(j) - 1] + length * sin(states[int(i), int(j)])
        line.set_data(x, y)
        return time_text, rect, line,


    anim = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
            interval=t[-1] / len(t) * 1000, blit=True, repeat=False)


    if filename is not None:
        anim.save('n_pendulum_pragram_2.gif', writer='pillow', fps=25)

animate_pendulum(t, y, arm_length, filename="n_pendulum_pragram_2.gif")


