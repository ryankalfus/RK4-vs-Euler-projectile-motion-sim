# --- imports ---
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML
from matplotlib.animation import FuncAnimation

# ----------------------------
# Projectile Motion Simulator
# Air resistance always ON (set by user)
# Compare integrators: RK4 vs Euler
# + 2D ball animation overlay with time overlay (mm:ss.hh)
# ----------------------------

# ---- Parameters you can change ----
g = 9.81                    # m/s^2
theta_deg = 45              # launch angle in degrees
v0 = 20.0                   # initial speed (m/s)
y0 = 0.0                    # initial height (m)

# Drag options:
use_quadratic_drag = True   # True: quadratic drag, False: linear drag

# Physical-ish defaults (good enough for a project)
m = 0.145                   # mass (kg) ~ baseball
rho = 1.225                 # air density (kg/m^3)
Cd = 0.47                   # drag coefficient (~sphere)
r = 0.0366                  # radius (m) ~ baseball
A = np.pi * r**2            # cross-sectional area (m^2)

# Quadratic drag coefficient: a_drag = -(k/m)*|v|*v
k_quad = 0.5 * rho * Cd * A

# Linear drag coefficient: a_drag = -(b/m)*v
b_lin = 0.02                # kg/s (tweak this if using linear drag)

dt = 0.1                    # time step (s)
t_max = 10.0                # max sim time (s)

# ---- Helper functions ----
def acceleration_with_drag(vx, vy):
    if use_quadratic_drag:
        v = np.hypot(vx, vy)
        ax = -(k_quad / m) * v * vx
        ay = -g - (k_quad / m) * v * vy
    else:
        ax = -(b_lin / m) * vx
        ay = -g - (b_lin / m) * vy
    return ax, ay

# ---- Integrators ----
def euler_step(state, dt, accel_func):
    x, y, vx, vy = state
    ax, ay = accel_func(vx, vy)

    x_new  = x  + dt * vx
    y_new  = y  + dt * vy
    vx_new = vx + dt * ax
    vy_new = vy + dt * ay

    return np.array([x_new, y_new, vx_new, vy_new], dtype=float)

def rk4_step(state, dt, accel_func):
    def f(s):
        x, y, vx, vy = s
        ax, ay = accel_func(vx, vy)
        return np.array([vx, vy, ax, ay], dtype=float)

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ---- Simulator ----
def simulate(accel_func, step_func):
    theta = np.deg2rad(theta_deg)
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    state = np.array([0.0, y0, vx0, vy0], dtype=float)

    t_vals = [0.0]
    x_vals = [state[0]]
    y_vals = [state[1]]
    vx_vals = [state[2]]
    vy_vals = [state[3]]

    t = 0.0
    for _ in range(int(t_max / dt)):
        state = step_func(state, dt, accel_func)
        t += dt

        if state[1] < 0 and t > dt:
            x_prev, y_prev = x_vals[-1], y_vals[-1]
            x_new, y_new = state[0], state[1]
            denom = (y_prev - y_new)
            frac = (y_prev / denom) if denom != 0 else 1.0

            x_land = x_prev + frac * (x_new - x_prev)
            t_land = t_vals[-1] + frac * (t - t_vals[-1])

            t_vals.append(t_land)
            x_vals.append(x_land)
            y_vals.append(0.0)
            vx_vals.append(state[2])
            vy_vals.append(state[3])
            break

        t_vals.append(t)
        x_vals.append(state[0])
        y_vals.append(state[1])
        vx_vals.append(state[2])
        vy_vals.append(state[3])

    return (np.array(t_vals), np.array(x_vals), np.array(y_vals),
            np.array(vx_vals), np.array(vy_vals))

# ---- Animation helpers ----
def format_mm_ss_hh(seconds_float):
    total_hundredths = int(round(seconds_float * 100))
    minutes = total_hundredths // (60 * 100)
    rem = total_hundredths % (60 * 100)
    secs = rem // 100
    hundredths = rem % 100
    return f"{minutes:02d}:{secs:02d}.{hundredths:02d}"

def overlay_ball_animation(t_rk, x_rk, y_rk, t_eu, x_eu, y_eu, title="RK4 vs Euler"):
    # Downsample each to ~400 frames
    def downsample(t, x, y, target=400):
        n = len(t)
        step = max(1, n // target)
        return t[::step], x[::step], y[::step]

    tR, xR, yR = downsample(t_rk, x_rk, y_rk)
    tE, xE, yE = downsample(t_eu, x_eu, y_eu)

    # Drive frames by the shorter one (keeps indexing safe)
    frames = min(len(tR), len(tE))

    fig, ax = plt.subplots()

    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    # Combined limits + padding
    x_max = float(max(np.max(xR), np.max(xE))) if frames else 1.0
    y_max = float(max(np.max(yR), np.max(yE))) if frames else 1.0
    ax.set_xlim(-0.05 * x_max, 1.05 * x_max)
    ax.set_ylim(-0.10 * max(1.0, y_max), 1.10 * max(1.0, y_max))

    # Ground line
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], linewidth=2)

    # Two trails + two balls (matplotlib picks different default colors)
    trail_rk, = ax.plot([], [], linewidth=2, alpha=0.35, label="RK4")
    trail_eu, = ax.plot([], [], linewidth=2, alpha=0.35, label="Euler")

    ball_rk, = ax.plot([], [], marker="o", markersize=14, label="RK4")
    ball_eu, = ax.plot([], [], marker="o", markersize=14, label="Euler")

    # Time overlay (top-left)
    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes,
        fontsize=14, va="top"
    )

    # Title overlay (top-center)
    title_text = ax.text(
        0.5, 0.98, title, transform=ax.transAxes,
        fontsize=14, va="top", ha="center"
    )

    # Leave space at bottom for a non-overlapping key
    fig.subplots_adjust(bottom=0.14)

    # Key at bottom-left (outside axes so it won't overlap animation)
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.10),   # bottom-left margin area
        frameon=True,
        ncol=1,
        handlelength=2.0,
        borderaxespad=0.0
    )
    legend.set_title("Key (color → method)")

    def init():
        trail_rk.set_data([], [])
        trail_eu.set_data([], [])
        ball_rk.set_data([], [])
        ball_eu.set_data([], [])
        time_text.set_text("")
        return trail_rk, trail_eu, ball_rk, ball_eu, time_text, title_text

    def update(i):
        trail_rk.set_data(xR[:i+1], yR[:i+1])
        trail_eu.set_data(xE[:i+1], yE[:i+1])

        ball_rk.set_data([xR[i]], [yR[i]])
        ball_eu.set_data([xE[i]], [yE[i]])

        # Use RK4 time for display (both use same dt typically, but this is safest)
        time_text.set_text(f"t = {format_mm_ss_hh(tR[i])}")
        return trail_rk, trail_eu, ball_rk, ball_eu, time_text, title_text

    ani = FuncAnimation(
        fig, update, frames=frames,
        init_func=init, interval=20, blit=True
    )

    plt.close(fig)
    return HTML(ani.to_jshtml())

# ==========================
# Run once: RK4 vs Euler (drag always ON)
# ==========================

clear_output(wait=True)

t_rk, x_rk, y_rk, vx_rk, vy_rk = simulate(acceleration_with_drag, rk4_step)
t_eu, x_eu, y_eu, vx_eu, vy_eu = simulate(acceleration_with_drag, euler_step)

def summarize(t_vals, x_vals, y_vals):
    return {
        "Range (m)": float(x_vals[-1]),
        "Max height (m)": float(np.max(y_vals)),
        "Time (s)": float(t_vals[-1]),
    }

stats_rk = summarize(t_rk, x_rk, y_rk)
stats_eu = summarize(t_eu, x_eu, y_eu)

print("Integrator comparison (same air resistance settings)")
print(f"RK4:   Range: {stats_rk['Range (m)']:.2f} m | Max height: {stats_rk['Max height (m)']:.2f} m | Time: {stats_rk['Time (s)']:.2f} s")
print(f"Euler: Range: {stats_eu['Range (m)']:.2f} m | Max height: {stats_eu['Max height (m)']:.2f} m | Time: {stats_eu['Time (s)']:.2f} s")

plt.figure()
plt.plot(x_rk, y_rk, label="RK4")
plt.plot(x_eu, y_eu, label="Euler")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Projectile Trajectory (RK4 vs Euler)")
plt.legend()
plt.grid(True)
plt.show()

speed_rk = np.hypot(vx_rk, vy_rk)
speed_eu = np.hypot(vx_eu, vy_eu)

plt.figure()
plt.plot(t_rk, speed_rk, label="RK4")
plt.plot(t_eu, speed_eu, label="Euler")
plt.xlabel("time (s)")
plt.ylabel("speed (m/s)")
plt.title("Speed vs Time (RK4 vs Euler)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t_rk, y_rk, label="RK4")
plt.plot(t_eu, y_eu, label="Euler")
plt.xlabel("time (s)")
plt.ylabel("height y (m)")
plt.title("Height vs Time (RK4 vs Euler)")
plt.legend()
plt.grid(True)
plt.show()

# Single overlay animation (RK4 + Euler)
display(overlay_ball_animation(t_rk, x_rk, y_rk, t_eu, x_eu, y_eu, "RK4 vs Euler"))

LAST_RESULTS = {
    "rk4": (t_rk, x_rk, y_rk, vx_rk, vy_rk),
    "euler": (t_eu, x_eu, y_eu, vx_eu, vy_eu),
}
