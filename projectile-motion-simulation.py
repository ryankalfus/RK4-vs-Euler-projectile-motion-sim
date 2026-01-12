# --- imports ---
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from matplotlib.animation import FuncAnimation

# ----------------------------
# Projectile Motion Simulator
# With and Without Air Resistance
# Toggle integrator (RK4/Euler) in Colab
# + 2D ball animation with time overlay (mm:ss.hh)
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

dt = 0.001                  # time step (s)
t_max = 10.0                # max sim time (s)

# ---- Helper functions ----
def acceleration_no_drag(vx, vy):
    return 0.0, -g

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
    # state = [x, y, vx, vy]
    x, y, vx, vy = state
    ax, ay = accel_func(vx, vy)

    x_new  = x  + dt * vx
    y_new  = y  + dt * vy
    vx_new = vx + dt * ax
    vy_new = vy + dt * ay

    return np.array([x_new, y_new, vx_new, vy_new], dtype=float)

def rk4_step(state, dt, accel_func):
    # state = [x, y, vx, vy]
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

        # Stop when projectile hits the ground (y <= 0) after launch
        if state[1] < 0 and t > dt:
            # Linear interpolation to estimate landing point nicely
            x_prev, y_prev = x_vals[-1], y_vals[-1]
            x_new, y_new = state[0], state[1]
            frac = y_prev / (y_prev - y_new) if (y_prev - y_new) != 0 else 1.0
            x_land = x_prev + frac * (x_new - x_prev)
            t_land = t_vals[-1] + frac * (t - t_vals[-1])

            # store landing point
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

# ---- Animation helper (2D ball + time overlay) ----
def format_mm_ss_hh(seconds_float):
    total_hundredths = int(round(seconds_float * 100))
    minutes = total_hundredths // (60 * 100)
    rem = total_hundredths % (60 * 100)
    secs = rem // 100
    hundredths = rem % 100
    return f"{minutes:02d}:{secs:02d}.{hundredths:02d}"

def trajectory_ball_animation(t_vals, x_vals, y_vals, title):
    # Downsample for speed (aim ~400 frames)
    n = len(t_vals)
    step = max(1, n // 400)
    t_ds = t_vals[::step]
    x_ds = x_vals[::step]
    y_ds = y_vals[::step]

    fig, ax = plt.subplots()

    # Show just the motion (no "graph look")
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    # Limits with padding
    x_max = float(np.max(x_ds)) if len(x_ds) else 1.0
    y_max = float(np.max(y_ds)) if len(y_ds) else 1.0
    ax.set_xlim(-0.05 * x_max, 1.05 * x_max)
    ax.set_ylim(-0.10 * max(1.0, y_max), 1.10 * max(1.0, y_max))

    # Ground line (comment this out if you want a totally blank background)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], linewidth=2)

    # Ball + (optional) faint trail
    trail, = ax.plot([], [], linewidth=2, alpha=0.35)
    ball, = ax.plot([], [], marker="o", markersize=14)

    # Time overlay (top-left)
    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes,
        fontsize=14, va="top"
    )

    # Title overlay (top-center) - optional
    title_text = ax.text(
        0.5, 0.98, title, transform=ax.transAxes,
        fontsize=14, va="top", ha="center"
    )

    def init():
        trail.set_data([], [])
        ball.set_data([], [])
        time_text.set_text("")
        return trail, ball, time_text, title_text

    def update(i):
        # trail up to i
        trail.set_data(x_ds[:i+1], y_ds[:i+1])
        # ball at i
        ball.set_data([x_ds[i]], [y_ds[i]])
        # time overlay
        time_text.set_text(f"t = {format_mm_ss_hh(t_ds[i])}")
        return trail, ball, time_text, title_text

    ani = FuncAnimation(
        fig, update, frames=len(t_ds),
        init_func=init, interval=20, blit=True
    )

    plt.close(fig)  # avoids duplicate static image in Colab
    return HTML(ani.to_jshtml())

# ==========================
# Colab dropdown toggle
# ==========================

method_dropdown = widgets.Dropdown(
    options=[("RK4", "rk4"), ("Euler", "euler")],
    value="rk4",
    description="Integrator:"
)

def run_selected_method(change=None):
    clear_output(wait=True)
    display(method_dropdown)

    # Choose integrator
    if method_dropdown.value == "euler":
        stepper = euler_step
        method_label = "Euler"
    else:
        stepper = rk4_step
        method_label = "RK4"

    # ---- Run simulations (both cases) ----
    t_n, x_n, y_n, vx_n, vy_n = simulate(acceleration_no_drag, stepper)
    t_d, x_d, y_d, vx_d, vy_d = simulate(acceleration_with_drag, stepper)

    # ---- Quick stats ----
    range_n = x_n[-1]
    range_d = x_d[-1]
    maxh_n = y_n.max()
    maxh_d = y_d.max()
    time_n = t_n[-1]
    time_d = t_d[-1]

    print(f"Integrator: {method_label}")
    print("No drag:")
    print(f"  Range: {range_n:.2f} m | Max height: {maxh_n:.2f} m | Time: {time_n:.2f} s")
    print("With drag:")
    print(f"  Range: {range_d:.2f} m | Max height: {maxh_d:.2f} m | Time: {time_d:.2f} s")

    # ---- Original plots ----
    # 1) Trajectory
    plt.figure()
    plt.plot(x_n, y_n, label="No air resistance")
    plt.plot(x_d, y_d, label="With air resistance")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"Projectile Trajectory ({method_label})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2) Speed vs time
    speed_n = np.hypot(vx_n, vy_n)
    speed_d = np.hypot(vx_d, vy_d)

    plt.figure()
    plt.plot(t_n, speed_n, label="No air resistance")
    plt.plot(t_d, speed_d, label="With air resistance")
    plt.xlabel("time (s)")
    plt.ylabel("speed (m/s)")
    plt.title(f"Speed vs Time ({method_label})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3) Height vs time
    plt.figure()
    plt.plot(t_n, y_n, label="No air resistance")
    plt.plot(t_d, y_d, label="With air resistance")
    plt.xlabel("time (s)")
    plt.ylabel("height y (m)")
    plt.title(f"Height vs Time ({method_label})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- New: 2D "ball" animations with time overlay ----
    display(trajectory_ball_animation(t_d, x_d, y_d, f"With Drag — {method_label}"))
    display(trajectory_ball_animation(t_n, x_n, y_n, f"No Drag — {method_label}"))

    global LAST_RESULTS
    LAST_RESULTS = {
        "method": method_label,
        "no_drag": (t_n, x_n, y_n),
        "drag": (t_d, x_d, y_d)
    }

# Re-run when dropdown changes
method_dropdown.observe(run_selected_method, names="value")
run_selected_method()
