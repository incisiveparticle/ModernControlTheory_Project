import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 140,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "lines.linewidth": 2.2
})


A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [12.0072944570369, -22.286266227076, 0, 0],
    [-14.4127962085308, 94.3805687203791, 0, 0]
])

Acl = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [217.402730766181, 97.4737535658231, 74.0314878330178, 25.9210907787797],
    [-815.96702676282, -362.037433275478, -288.472088625334, -99.9308723606861]
])

Bf = np.array([
    [0],
    [0],
    [-0.0185452297547909],
    [0.173775671406003]
])

Klqr = np.array([
    [659.369881335997, 185.814197828964, 229.759201281148, 60.9611748853657],
    [652.303881844964, 332.73003553997, 233.219053424557, 76.9882420218239]
])

# Initial condition

x0 = np.array([5*np.pi/180, -3*np.pi/180, 0, 0])


def open_loop_dynamics(t, x):
    return A @ x

def closed_loop_dynamics(t, x):
    return Acl @ x

def external_force(t):
    return 20.0 if 1.0 <= t <= 1.5 else 0.0

def disturbed_dynamics(t, x):
    return Acl @ x + (Bf.flatten() * external_force(t))

def constant_force(t):
    return 20.0
def constant_force_dynamics(t, x):
    return Acl @ x + (Bf.flatten() * constant_force(t))
# Simulations

# Open-loop
t_ol = np.linspace(0, 0.25, 2000)
sol_ol = solve_ivp(open_loop_dynamics, [0, 0.25], x0, t_eval=t_ol)

# Closed-loop
t_cl = np.linspace(0, 5, 3000)
sol_cl = solve_ivp(closed_loop_dynamics, [0, 5], x0, t_eval=t_cl)

# Disturbance response
t_d = np.linspace(0, 8, 4000)
sol_d = solve_ivp(disturbed_dynamics, [0, 8], x0, t_eval=t_d)

x_d = sol_d.y.T
Fext = np.array([external_force(t) for t in t_d])

# Constant force response
t_cf = np.linspace(0, 8, 4000)
sol_cf = solve_ivp(constant_force_dynamics, [0, 8], x0, t_eval=t_cf)

x_cf = sol_cf.y.T
Fconst = np.array([constant_force(t) for t in t_cf])

# Control torques under constant force
u_control_cf = np.array([-(Klqr @ x) for x in x_cf])

# Control torques
u_control = np.array([-(Klqr @ x) for x in x_d])

# Figure 1: Open-loop response

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(sol_ol.t, sol_ol.y[0]*180/np.pi, color='red')
axs[0].set_ylabel(r'$\theta_1$ (deg)')
axs[0].set_title('Open-loop Response: Ankle Angle')

axs[1].plot(sol_ol.t, sol_ol.y[1]*180/np.pi, color='red')
axs[1].set_ylabel(r'$\theta_2$ (deg)')
axs[1].set_xlabel('Time (s)')
axs[1].set_title('Open-loop Response: Hip Relative Angle')

plt.tight_layout()
# plt.savefig("open_loop_response.png", bbox_inches='tight')


# Figure 2: Closed-loop response

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(sol_cl.t, sol_cl.y[0]*180/np.pi, color='blue')
axs[0].set_ylabel(r'$\theta_1$ (deg)')
axs[0].set_title('Closed-loop Response: Ankle Angle')

axs[1].plot(sol_cl.t, sol_cl.y[1]*180/np.pi, color='blue')
axs[1].set_ylabel(r'$\theta_2$ (deg)')
axs[1].set_xlabel('Time (s)')
axs[1].set_title('Closed-loop Response: Hip Relative Angle')

plt.tight_layout()
# plt.savefig("closed_loop_response.png", bbox_inches='tight')

# Figure 3: Disturbance response + force

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))

axs[0].plot(t_d, x_d[:,0]*180/np.pi, color='tab:green')
axs[0].set_ylabel(r'$\theta_1$ (deg)')
axs[0].set_title('Closed-loop Response under External Push: Ankle Angle')

axs[1].plot(t_d, x_d[:,1]*180/np.pi, color='tab:green')
axs[1].set_ylabel(r'$\theta_2$ (deg)')
axs[1].set_title('Closed-loop Response under External Push: Hip Relative Angle')

axs[2].plot(t_d, Fext, color='black')
axs[2].set_ylabel(r'$F_{ext}$ (N)')
axs[2].set_xlabel('Time (s)')
axs[2].set_title('Applied External Force Profile')

plt.tight_layout()
# plt.savefig("disturbance_response.png", bbox_inches='tight')

# Figure 4: Control torques

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(t_d, u_control[:,0], color='purple')
axs[0].set_ylabel(r'$\tau_1$ (Nm)')
axs[0].set_title('Control Torque at Ankle')

axs[1].plot(t_d, u_control[:,1], color='purple')
axs[1].set_ylabel(r'$\tau_2$ (Nm)')
axs[1].set_xlabel('Time (s)')
axs[1].set_title('Control Torque at Hip')

plt.tight_layout()
# plt.savefig("control_torques.png", bbox_inches='tight')


max_theta1 = np.max(np.abs(x_d[:,0])) * 180/np.pi
max_theta2 = np.max(np.abs(x_d[:,1])) * 180/np.pi
max_tau1 = np.max(np.abs(u_control[:,0]))
max_tau2 = np.max(np.abs(u_control[:,1]))

print('------------------------------------')
print(f'Maximum ankle angle under disturbance = {max_theta1:.4f} deg')
print(f'Maximum hip relative angle under disturbance = {max_theta2:.4f} deg')
print(f'Maximum ankle torque required = {max_tau1:.4f} Nm')
print(f'Maximum hip torque required   = {max_tau2:.4f} Nm')



# Figure 5: Constant force response

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))

axs[0].plot(t_cf, x_cf[:,0]*180/np.pi, color='darkorange')
axs[0].set_ylabel(r'$\theta_1$ (deg)')
axs[0].set_title('Closed-loop Response under Constant External Force: Ankle Angle')

axs[1].plot(t_cf, x_cf[:,1]*180/np.pi, color='darkorange')
axs[1].set_ylabel(r'$\theta_2$ (deg)')
axs[1].set_title('Closed-loop Response under Constant External Force: Hip Relative Angle')

axs[2].plot(t_cf, Fconst, color='black')
axs[2].set_ylabel(r'$F_{ext}$ (N)')
axs[2].set_xlabel('Time (s)')
axs[2].set_title('Constant External Force Profile')

plt.tight_layout()
# plt.savefig("constant_force_response.png", bbox_inches='tight')

# Figure 6: Control torques under constant force

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(t_cf, u_control_cf[:,0], color='teal')
axs[0].set_ylabel(r'$\tau_1$ (Nm)')
axs[0].set_title('Control Torque at Ankle under Constant Force')

axs[1].plot(t_cf, u_control_cf[:,1], color='teal')
axs[1].set_ylabel(r'$\tau_2$ (Nm)')
axs[1].set_xlabel('Time (s)')
axs[1].set_title('Control Torque at Hip under Constant Force')

plt.tight_layout()
# plt.savefig("constant_force_torques.png", bbox_inches='tight')

max_theta1_cf = np.max(np.abs(x_cf[:,0])) * 180/np.pi
max_theta2_cf = np.max(np.abs(x_cf[:,1])) * 180/np.pi
ss_theta1_cf = x_cf[-1,0] * 180/np.pi
ss_theta2_cf = x_cf[-1,1] * 180/np.pi
ss_tau1_cf = u_control_cf[-1,0]
ss_tau2_cf = u_control_cf[-1,1]

print('------------------------------------')
print('Constant Force Case:')
print(f'Maximum ankle angle = {max_theta1_cf:.4f} deg')
print(f'Maximum hip relative angle = {max_theta2_cf:.4f} deg')
print(f'Steady-state ankle angle = {ss_theta1_cf:.4f} deg')
print(f'Steady-state hip angle   = {ss_theta2_cf:.4f} deg')
print(f'Steady-state ankle torque = {ss_tau1_cf:.4f} Nm')
print(f'Steady-state hip torque   = {ss_tau2_cf:.4f} Nm')

plt.show()