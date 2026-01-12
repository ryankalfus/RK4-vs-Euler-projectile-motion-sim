# projectile-motion-simulation
A Python-based numerical simulation of projectile motion with optional air resistance and interactive visualization.

# Numerical Projectile Motion Simulation

This repository contains a Python-based numerical simulation of two-dimensional projectile motion, comparing ideal motion (no air resistance) with motion including linear and quadratic drag.

The project focuses on using numerical methods to model realistic physical systems that do not admit simple closed-form solutions.

---

## Overview

The simulation models a projectile launched at a given angle and speed under gravity. Two physical cases are simulated:

* **Ideal projectile motion** (gravity only)
* **Projectile motion with air resistance**, using either linear or quadratic drag models

The equations of motion are solved numerically using either:

* **Fourth-order Runge–Kutta (RK4)** integration, or  
* **Forward Euler integration**

The integration method can be selected interactively in a Jupyter or Google Colab environment.

---

## Features

* Ideal (no-drag) projectile motion
* Air resistance models:
  * Linear drag
  * Quadratic drag
* Adjustable physical parameters:
  * Mass
  * Drag coefficients
  * Launch angle
  * Initial speed
  * Timestep
* Multiple numerical integration methods:
  * Fourth-order Runge–Kutta (RK4)
  * Forward Euler
* Interactive integrator selection via dropdown (Jupyter / Colab)
* Automatic detection of ground impact with interpolation for accurate landing position
* Static visualizations:
  * Trajectory (x–y)
  * Speed vs. time
  * Height vs. time
* Animated visualization:
  * 2D projectile motion (“ball” animation)
  * Fading trajectory trail
  * Time overlay formatted as **mm:ss.hh**

---

## Methods

* Newton’s Second Law applied to two-dimensional motion
* System of coupled first-order ordinary differential equations
* Numerical time integration using:
  * Fourth-order Runge–Kutta (RK4)
  * Forward Euler method
* Fixed timestep integration
* Event detection for ground impact using linear interpolation
* Numerical comparison between integration schemes

---

## Requirements

* Python 3
* NumPy
* Matplotlib
* ipywidgets (for interactive controls)
* Jupyter Notebook or Google Colab (recommended)

---

## Usage

1. Open the notebook in Jupyter or Google Colab.
2. Adjust physical and numerical parameters in the configuration section.
3. Use the dropdown menu to select the numerical integrator (RK4 or Euler).
4. Run the notebook to:
   * Compare trajectories with and without air resistance
   * Visualize numerical differences between integration methods
   * View both static plots and animated projectile motion

---

## Possible Extensions

* Adaptive timestep methods
* Energy conservation and numerical error analysis
* Additional integrators (Verlet, midpoint, symplectic methods)
* Quantitative comparison of Euler vs RK4 error
* Experimental validation with real projectile data
