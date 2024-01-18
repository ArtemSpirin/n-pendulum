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

q = dynamicsymbols('q:' + str(n + 1))  # Generalized coordinates
u = dynamicsymbols('u:' + str(n + 1))  # Generalized speeds
f = dynamicsymbols('f')                # Force applied to the cart

m = symbols('m:' + str(n + 1))         # Mass of each bob
l = symbols('l:' + str(n))             # Length of each link
g, t = symbols('g t')



I = ReferenceFrame('I')                # Inertial reference frame
O = Point('O')                         # Origin point
O.set_vel(I, 0)                        # Origin's velocity is zero

P0 = Point('P0')                       # Hinge point of top link
P0.set_pos(O, q[0] * I.x)              # Set the position of P0
P0.set_vel(I, u[0] * I.x)              # Set the velocity of P0
Pa0 = Particle('Pa0', P0, m[0])        # Define a particle at P0


frames = [I]                              # List to hold the n + 1 frames
points = [P0]                             # List to hold the n + 1 points
particles = [Pa0]                         # List to hold the n + 1 particles
forces = [(P0, f * I.x - m[0] * g * I.y)] # List to hold the n + 1 applied forces, including the input force, f
kindiffs = [q[0].diff(t) - u[0]]          # List to hold kinematic ODE's

for i in range(n):
    Bi = I.orientnew('B' + str(i), 'Axis', [q[i + 1], I.z])   # Create a new frame
    Bi.set_ang_vel(I, u[i + 1] * I.z)                         # Set angular velocity
    frames.append(Bi)                                         # Add it to the frames list

    Pi = points[-1].locatenew('P' + str(i + 1), l[i] * Bi.x)  # Create a new point
    Pi.v2pt_theory(points[-1], I, Bi)                         # Set the velocity
    points.append(Pi)                                         # Add it to the points list

    Pai = Particle('Pa' + str(i + 1), Pi, m[i + 1])           # Create a new particle
    particles.append(Pai)                                     # Add it to the particles list

    forces.append((Pi, -m[i + 1] * g * I.y))                  # Set the force applied at the point

    kindiffs.append(q[i + 1].diff(t) - u[i + 1])              # Define the kinematic ODE:  dq_i / dt - u_i = 0


kane = KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs) # Initialize the object
fr, frstar = kane.kanes_equations(particles, forces)    # Generate EoM's fr + frstar = 0

arm_length = 1. / n  # The maximum length of the pendulum is 1 meter
bob_mass = 0.0001 / n  # The maximum mass of the bobs is 10 grams
parameters = [g, m[0]]  # Parameter definitions starting with gravity and the first bob
parameter_vals = [9.81, 5 / n]  # Numerical values for the first two
for i in range(n):  # Then each mass and length
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




dynamic = q + u                                                # Make a list of the states
dynamic.append(f)                                              # Add the input force
dummy_symbols = [Dummy() for i in dynamic]                     # Create a dummy symbol for each variable
dummy_dict = dict(zip(dynamic, dummy_symbols))
kindiff_dict = kane.kindiffdict()                              # Get the solved kinematical differential equations
M = kane.mass_matrix_full.subs(kindiff_dict).subs(dummy_dict)  # Substitute into the mass matrix
F = kane.forcing_full.subs(kindiff_dict).subs(dummy_dict)      # Substitute into the forcing vector
M_func = lambdify(dummy_symbols + parameters, M)               # Create a callable function to evaluate the mass matrix
F_func = lambdify(dummy_symbols + parameters, F)               # Create a callable function to evaluate the forcing vector


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
    u = 0  # The input force is always zero
    arguments = hstack((x, u, args))  # States, input, and parameters
    dx = array(solve(M_func(*arguments),  # Solving for the derivatives
                     F_func(*arguments))).T[0]

    return dx

x0 = hstack(( 0, -3*pi / 8 * ones(len(q) - 1), 1e-3 * ones(len(u)) )) # Initial conditions, q and u
t = linspace(0, 15, 1000)                                          # Time vector
y = odeint(right_hand_side, x0, t, args=(parameter_vals,)) # Actual integration
y_s   =  list()
for i in range(len(y)):
    y_s.append(y[i][0])

fig, ax = plt.subplots()
ax.plot(t, y_s)
plt.show()

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
    # the number of pendulum bobs
    numpoints = states.shape[1] / 2

    # first set up the figure, the axis, and the plot elements we want to animate
    fig = plt.figure()

    # some dimesions
    cart_width = 0.4
    cart_height = 0.2

    # set the limits based on the motion
    xmin = around(states[:, 0].min() - cart_width / 2.0, 1)-1
    xmax = around(states[:, 0].max() + cart_width / 2.0, 1)+1

    # create the axes
    ax = plt.axes(xlim=(xmin, xmax), ylim=(-1.1, 1.1), aspect='equal')

    # display the current time
    time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)

    # create a rectangular cart
    rect = Rectangle([states[0, 0] - cart_width / 2.0, -cart_height / 2],
        cart_width, cart_height, fill=True, color='red', ec='black')
    ax.add_patch(rect)

    # blank line for the pendulum
    line, = ax.plot([], [], lw=2, marker='o', markersize=6)

    # initialization function: plot the background of each frame
    def init():
        time_text.set_text('')
        rect.set_xy((0.0, 0.0))
        line.set_data([], [])
        return time_text, rect, line,

    # animation function: update the objects
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

    # call the animator function
    anim = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
            interval=t[-1] / len(t) * 1000, blit=True, repeat=False)

    # save the animation if a filename is given
    if filename is not None:
        anim.save('pen.gif', writer='pillow', fps=25)

animate_pendulum(t, y, arm_length, filename="pen.gif")


