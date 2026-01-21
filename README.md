# Numerical Projectile Motion Simulation: RK4 vs Euler (with Optional Closed-Form Reference)

A Python-based numerical simulation of 2D projectile motion with visualization comparing **RK4** and **Euler** time-stepping. The simulation supports **quadratic drag** (default) and a **no-drag mode**. When drag is turned **off**, the project additionally plots the **closed-form analytical solution** as a third curve for direct validation of the numerical integrators.

---

## Overview

This project simulates two-dimensional projectile motion and compares two numerical integration methods on the **same physical system** (same initial conditions, same timestep, same force model):

- **Fourth-order Runge–Kutta (RK4)**
- **Forward Euler integration**

### Two physics modes

- **Quadratic drag ON** (`use_quadratic_drag = True`)
  - The motion is non-linear and (in general) **does not have a closed-form solution** for position vs time.
  - Only RK4 and Euler are shown.

- **No drag** (`use_quadratic_drag = False`)
  - The motion reduces to standard constant-acceleration projectile motion with a **closed-form analytical solution**.
  - RK4 and Euler are shown **alongside** the analytical solution as a third curve.

---

## Features

### Physics and simulation
- 2D projectile motion under gravity
- Toggleable drag model:
  - **Quadratic drag** (default)
  - **No drag** (set `use_quadratic_drag = False`)
- Fixed timestep integration
- Automatic ground-impact detection with **linear interpolation** for more accurate landing time/range

### Numerical methods
- Forward Euler (first order)
- RK4 (fourth order)

### Visualizations
Static plots always included:
- **Trajectory** (x vs y)
- **Speed vs time**
- **Height vs time**

Additional analytical validation (only when drag is OFF):
- The above three plots include a **third curve**: **Analytical (closed-form)**

Animated overlay visualization:
- 2D animation with fading trails and mm:ss.hh time overlay
- Shows:
  - RK4 vs Euler when drag is ON
  - RK4 vs Euler vs Analytical when drag is OFF

### Error analysis (numerical self-consistency)
- **Error vs time**: position error relative to a higher-resolution RK4 reference run
- **Error vs timestep size** (log-log): maximum position error over the run as `dt` is varied

---

## Methods

- Newton’s Second Law in 2D
- ODE system in first-order form:
  - state = (x, y, vx, vy)
- Integration methods:
  - Forward Euler
  - RK4
- Force model:
  - Quadratic drag (if enabled): acceleration depends on speed and direction
  - No drag (if disabled): acceleration is constant (ax = 0, ay = −g)
- Ground-impact event detection:
  - Stops when y crosses below 0
  - Interpolates between the last two points to estimate landing time/range more accurately

---

## Why the analytical curve only appears with drag OFF

Projectile motion **without drag** has a standard closed-form solution for:
- x(t), y(t)
- vx(t), vy(t)

However, when **quadratic drag** is enabled, the equations become **nonlinear** and generally **do not have a closed-form solution** for position vs time. Because of this, the analytical curve is only plotted and animated when drag is disabled (`use_quadratic_drag = False`).

---

## Requirements

- Python 3
- NumPy
- Matplotlib
- Jupyter Notebook or Google Colab (recommended)

*(If you run this inside a notebook environment, the animation display is embedded.)*

---

## Usage

1. Open the notebook in Jupyter or Google Colab.
2. Set parameters near the top:
   - `theta_deg`, `v0`, `y0`, `dt`, etc.
3. Choose the physics mode:
   - Drag ON: `use_quadratic_drag = True`
   - Drag OFF (enables analytical reference): `use_quadratic_drag = False`
4. Run the notebook to generate:
   - Trajectory, speed vs time, height vs time plots
   - Error plots (error vs time, error vs timestep size)
   - Overlay animation

---

## Interpreting results

- With **drag ON**, RK4 and Euler will diverge increasingly over time because Euler accumulates more numerical error and is less stable for nonlinear dynamics.
- With **drag OFF**, the **analytical closed-form curve** provides a direct correctness check:
  - RK4 should track very closely for moderate dt
  - Euler will deviate noticeably unless dt is made small

The error plots provide quantitative evidence of accuracy and timestep sensitivity.

---

## Possible Extensions

- Add additional integrators (midpoint, Verlet, symplectic methods)
- Estimate convergence slopes from the log-log timestep plot
- Add adaptive timestep control
- Compare against experimental data
- Track conserved quantities in the no-drag case (e.g., mechanical energy) and quantify drift

---
