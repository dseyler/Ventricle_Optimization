#!/usr/bin/env python3
"""
Generate a .dat file with a linear pressure ramp:
- 0 Pa at t=0
- 6894.76 Pa at t=0.5
- 0 Pa at t=1
First row: number of time steps (1001) and number of fourier modes (256)
Each subsequent row: <time> <pressure>
Output format matches stress.dat
"""

import numpy as np

def generate_pressure_dat(filename="pressure.dat", n_steps=1001, n_modes=256, max_pressure=6894.76 , total_time=0.3, ramp_up_time=0.1):
    # Time array
    t = np.linspace(0, total_time, n_steps)
    pressure = np.zeros_like(t)
    
    # Linear ramp up to t=0.5, then ramp down
    for i, ti in enumerate(t):
        if ti <= ramp_up_time:
            pressure[i] = (ti / ramp_up_time) * max_pressure
        else:
            pressure[i] = max_pressure #((total_time - ti) / (total_time - ramp_up_time)) * max_pressure
    
    # Write to file
    with open(filename, 'w') as f:
        f.write(f"{n_steps} {n_modes}\n")
        for ti, p in zip(t, pressure):
            f.write(f"{ti:.6f} {p:.6f}\n")
    print(f"Pressure data written to {filename}")

if __name__ == "__main__":
    generate_pressure_dat()
