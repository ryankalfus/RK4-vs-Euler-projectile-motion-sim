# projectile-motion-simulation
A Python-based numerical simulation of projectile motion with air resistance.
# Numerical Projectile Motion Simulation

This repository contains a Python-based numerical simulation of two-dimensional projectile motion, comparing ideal motion (no air resistance) with motion including linear and quadratic drag.

The project focuses on using numerical methods to model realistic physical systems that do not admit simple closed-form solutions.

---

## Overview

The simulation models a projectile launched at a given angle and speed under gravity. Two cases are simulated:

* **Ideal projectile motion** (gravity only)
* **Projectile motion with air resistance**, using either linear or quadratic drag models

A fourth-order Runge–Kutta (RK4) integrator is used to solve the coupled differential equations governing position and velocity.

---

## Features

* Ideal (no-drag) projectile motion
* Air resistance models
* Adjustable physical parameters (mass, drag coefficient, timestep, launch angle)
* Numerical integration using RK4
* Automatic detection of ground impact
* Visualization of:

  * Trajectory (x–y)
  * Speed vs. time
  * Height vs. time

---

## Motivation

This project was developed as an independent computational physics exercise while studying AP Physics 1.
The goal was to explore how numerical methods can be used to model realistic motion and to study the effects of air resistance on projectile trajectories.

---

## Methods

* Newton’s Second Law applied to 2D motion
* Systems of first-order differential equations
* Fourth-order Runge–Kutta (RK4) time-stepping
* Event detection via interpolation for accurate landing position

---

## Requirements

* Python 3
* NumPy
* Matplotlib

---

## Usage

Run the simulation in a Jupyter notebook or Google Colab environment.
Modify parameters in the configuration section to explore different launch conditions and drag models.

---

## Possible Extensions

* Adaptive timestep methods
* Energy analysis and numerical error estimation
* Comparison with Euler or Verlet integration
* Experimental validation with real projectile data
