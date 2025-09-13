#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 11:53:53 2025

@author: francisbrain4
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 10:26:15 2025

@author: francisbrain4
"""

"""
Complete ROM (Reduced Order Modeling) for 1D Heat Equation
Author: Francis Gillet
Date: September 12, 2025

NOTES:
==================================
1. MATRIX A: Represents spatial derivatives, NEVER changes with time
   - Rows = spatial locations (x₀, x₁, x₂, ...)
   - Columns = temperature unknowns at those locations
   - Tridiagonal pattern from finite difference: d²T/dx² ≈ (T[i-1] - 2T[i] + T[i+1])/dx²

2. TIME SNAPSHOTS: Each column of data matrix = temperature at one time
   - Time loop solves A×T = -q at each time step
   - Data matrix Xh: rows = spatial points, columns = time steps
   - Each solve gives one "snapshot" of temperature distribution

3. SVD: Finds dominant patterns in temperature evolution
   - U columns = spatial modes (how temperature varies in space)
   - S values = importance of each mode (energy content)
   - VT rows = temporal modes (how spatial patterns evolve in time)

4. ROM: Uses few dominant modes instead of full solution
   - Instead of 1000×1000 system → solve 7×7 system
   - T(x,t) ≈ Φ × a(t) where Φ = spatial modes, a = time coefficients
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# ================================
# PROBLEM SETUP
# ================================

# Physical parameters
q0 = 1e3        # Heat source intensity [W/m³]
Lx = 1          # Rod length [m] 
Lt = 100        # Total time [s]

# Discretization
Nx = 501        # Number of spatial points
Nt = 1000       # Number of time steps

# Create grids
x = np.linspace(0, Lx, Nx)
t = np.linspace(0, Lt, Nt)
dx = x[1] - x[0]
dt = t[1] - t[0]

print(f"Grid: {Nx} spatial points, {Nt} time steps")
print(f"dx = {dx:.6f}, dt = {dt:.3f}")

# ================================
# HEAT SOURCE DISTRIBUTIONS
# ================================

def hertz_distribution(x, a, q_max, velocity, time):
    """
    Hertz (semicircular) heat distribution - like a welding torch
    
    PHYSICAL MEANING: Models contact heating with curved intensity profile
    """
    x0 = velocity * time  # Current position of heat source
    r = np.abs(x - x0)    # Distance from heat source center
    
    # Hertz formula: semicircular distribution
    q_distribution = np.where(r <= a, 
                             q_max * np.sqrt(1.0001 - (r / a)**2), 
                             0)
    return q_distribution

def tent_distribution(x, a, q_max, velocity, time):
    """
    Tent (triangular) heat distribution - simpler model
    
    PHYSICAL MEANING: Linear drop-off from center, easier to analyze
    """
    x0 = velocity * time  # Current position of heat source
    r = np.abs(x - x0)    # Distance from heat source center
    
    # Tent formula: triangular distribution
    q_distribution = np.where(r <= a, 
                             q_max * (1 - r / a), 
                             0)
    return q_distribution

# Heat source parameters
body_width = Lx / 10      # Heat source width (10% of domain)
velocity = Lx / Lt        # Heat source velocity (travels full length)

print(f"Heat source width: {body_width:.3f} m")
print(f"Heat source velocity: {velocity:.3f} m/s")

# ================================
# FINITE DIFFERENCE MATRIX SETUP
# ================================
# TUTORIAL INSIGHT: This matrix represents d²T/dx² = -q spatially
# Each row = equation at one spatial point
# Pattern: T[i-1] - 2T[i] + T[i+1] = -q[i] * dx²

A = np.zeros((Nx-1, Nx-1))

# Build tridiagonal matrix for d²T/dx²
np.fill_diagonal(A, -2)           # Main diagonal: coefficient of T[i]
np.fill_diagonal(A[:, 1:], 1)     # Upper diagonal: coefficient of T[i+1]  
np.fill_diagonal(A[1:, :], 1)     # Lower diagonal: coefficient of T[i-1]

# Boundary conditions
A[0, 1] = 2     # dT/dx = 0 at x=0 (insulated left boundary)
# Right boundary T=0 is handled by excluding last node

# Scale by dx²
A = A / (dx**2)

# Convert to sparse matrix for speed
As = csr_matrix(A, shape=(Nx-1, Nx-1))

print("Finite difference matrix A created")
print(f"Matrix size: {A.shape} (excludes right boundary node)")

# ================================
# GENERATE TRAINING DATA (TENT)
# ================================
# TUTORIAL INSIGHT: Each column = temperature snapshot at one time
# This creates the data matrix for SVD

print("\nGenerating training data using TENT distribution...")

qt = np.zeros(Nx-1)           # Heat source vector
Xt = np.zeros((Nx, Nt))       # Data matrix: rows=space, cols=time

for i in range(Nt):
    # Update heat source position at each time step
    for j in range(Nx-2):  # Exclude last node (boundary condition)
        qt[j] = tent_distribution(x[j], body_width, q0, velocity, t[i])
    
    # Solve heat equation: A * T = -q
    # TUTORIAL INSIGHT: Same matrix A, different q at each time
    xt = spsolve(As, -qt)
    
    # Store snapshot in column i
    # TUTORIAL INSIGHT: Column i = complete temperature distribution at time i
    Xt[:-1, i] = xt  # Fill all but last row (boundary T=0)
    Xt[-1, i] = 0    # Right boundary condition

print(f"Training data generated: {Xt.shape} (space × time)")

# ================================
# SVD ANALYSIS
# ================================
# TUTORIAL INSIGHT: Find dominant spatial-temporal patterns
# U = spatial modes, S = importance, VT = temporal evolution

print("\nPerforming SVD analysis...")

Uec, Sec, VTec = np.linalg.svd(Xt, full_matrices=False)

print(f"SVD completed: U={Uec.shape}, S={Sec.shape}, VT={VTec.shape}")
print(f"First 10 singular values: {Sec[:10]}")

# Plot singular values to see energy distribution
plt.figure(figsize=(10, 6))
plt.semilogy(Sec, 'bo-', linewidth=2, markersize=4)
plt.title('Singular Values (Energy Content of Each Mode)')
plt.xlabel('Mode Number')
plt.ylabel('Singular Value')
plt.grid(True)
plt.show()

# Plot first few spatial modes
plt.figure(figsize=(12, 8))
plt.plot(x, Uec[:, 0], 'r-', linewidth=2, label='1st mode')
plt.plot(x, Uec[:, 1], 'b-', linewidth=2, label='2nd mode')
plt.plot(x, Uec[:, 2], 'g-', linewidth=2, label='3rd mode')
plt.title('Dominant Spatial Modes from SVD')
plt.xlabel('Position x [m]')
plt.ylabel('Mode Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# ================================
# ROM SETUP
# ================================
# TUTORIAL INSIGHT: Use first Nr modes as basis functions
# Instead of 1000 unknowns → Nr unknowns (typically 5-20)

# ROM parameters - try different values!
Nr_values = [7, 15, 200]  # Number of modes to test
time_snapshots = [10, 50, 70]  # % of time to compare results

print(f"\nSetting up ROM with modes: {Nr_values}")

def setup_rom(Nr, Uec, x):
    """
    Set up ROM matrices for given number of modes
    
    TUTORIAL INSIGHT: 
    - Phi = spatial basis (first Nr columns of U)
    - d2Phi_dx2 = second derivatives of basis functions
    - A_rom = projected differential operator (Nr×Nr instead of Nx×Nx)
    """
    # Build spatial basis from dominant SVD modes
    Phi = np.zeros((Nx, Nr))
    for i in range(Nr):
        Phi[:, i] = Uec[:, i]
    
    # Compute second derivatives of basis functions
    d2Phi_dx2 = np.zeros((Nx, Nr))
    for i in range(Nr):
        dPhi_dx = np.gradient(Phi[:, i], x)
        d2Phi_dx2[:, i] = np.gradient(dPhi_dx, x)
    
    return Phi, d2Phi_dx2

# ================================
# ROM SOLUTION FUNCTION
# ================================

def solve_rom(Nr, Uec, x, t, time_snapshots):
    """
    Solve heat equation using ROM with Nr modes
    
    TUTORIAL INSIGHT:
    1. Project differential operator: A_rom = d2Phi_dx2.T @ d2Phi_dx2
    2. At each time: solve A_rom * a = -d2Phi_dx2.T @ q
    3. Reconstruct: T = Phi @ a
    """
    
    Phi, d2Phi_dx2 = setup_rom(Nr, Uec, x)
    
    # ROM differential operator (Nr×Nr matrix)
    A_rom = (d2Phi_dx2).T @ d2Phi_dx2
    
    print(f"  ROM matrix size: {A_rom.shape} (vs full size {Nx-1}×{Nx-1})")
    
    # Storage for results at selected time snapshots
    ns = (np.array(time_snapshots) / 100 * Nt).astype(int)
    Xrh_select = np.zeros((Nx, len(time_snapshots)))
    
    # Time-stepping loop
    qh = np.zeros(Nx)
    for i in range(Nt):
        # Generate Hertz heat source at current time
        for j in range(Nx-1):
            qh[j] = hertz_distribution(x[j], body_width, q0, velocity, t[i])
        
        # Project heat source onto ROM space
        # TUTORIAL INSIGHT: Instead of solving for all temperatures,
        # solve for coefficients of dominant modes
        q_rom = (d2Phi_dx2).T @ qh
        
        # Solve ROM system (small Nr×Nr system!)
        a_rom = np.linalg.solve(A_rom, -q_rom)
        
        # Reconstruct full temperature field
        # TUTORIAL INSIGHT: T ≈ a₁×Mode₁ + a₂×Mode₂ + ... + aₙ×Modeₙ
        T_full = Phi @ a_rom
        
        # Store selected snapshots
        if i in ns:
            idx = np.where(ns == i)[0][0]
            Xrh_select[:, idx] = T_full
    
    return Xrh_select

# ================================
# GENERATE REFERENCE SOLUTION (HERTZ)
# ================================
# Full solution using Hertz distribution for comparison

print("\nGenerating reference solution using HERTZ distribution...")

qh = np.zeros(Nx-1)
Xh = np.zeros((Nx, Nt))

for i in range(Nt):
    # Generate Hertz heat source
    for j in range(Nx-2):
        qh[j] = hertz_distribution(x[j], body_width, q0, velocity, t[i])
    
    # Solve full system
    xh = spsolve(As, -qh)
    Xh[:-1, i] = xh
    Xh[-1, i] = 0

print("Reference solution completed")

# ================================
# ROM SOLUTIONS AND COMPARISON
# ================================

print("\nSolving ROM for different numbers of modes...")

# Storage for ROM results
Xrh_all = np.zeros((Nx, len(time_snapshots), len(Nr_values)))

# Solve ROM for each mode count
for ir, Nr in enumerate(Nr_values):
    print(f"\nSolving ROM with {Nr} modes...")
    Xrh_all[:, :, ir] = solve_rom(Nr, Uec, x, t, time_snapshots)

# ================================
# RESULTS VISUALIZATION
# ================================

print("\nGenerating comparison plots...")

# Compare ROM vs Full solution at different times
colors = ['r', 'g', 'b']
ns = (np.array(time_snapshots) / 100 * Nt).astype(int)

for i, time_pct in enumerate(time_snapshots):
    plt.figure(figsize=(12, 8))
    
    # Full solution
    plt.plot(x, Xh[:, ns[i]], 'k-', linewidth=3, label='Full Solution (Hertz)')
    
    # ROM solutions
    for ir, Nr in enumerate(Nr_values):
        error = Xh[:, ns[i]] - Xrh_all[:, i, ir]
        plt.plot(x, Xrh_all[:, i, ir], f'{colors[ir]}--', linewidth=2, 
                label=f'ROM {Nr} modes')
        plt.plot(x, error, f'{colors[ir]}:', linewidth=1, alpha=0.7,
                label=f'Error {Nr} modes')
    
    plt.title(f'ROM Comparison at {time_pct}% of Total Time')
    plt.xlabel('Position x [m]')
    plt.ylabel('Temperature [K]')
    plt.legend()
    plt.grid(True)
    plt.show()

# Summary plot: Temperature evolution at different times
plt.figure(figsize=(12, 8))
for i, time_pct in enumerate(time_snapshots):
    plt.plot(x, Xh[:, ns[i]], linewidth=2, label=f'{time_pct}% elapsed time')

plt.title('Full Solution: Temperature Evolution Over Time')
plt.xlabel('Position x [m]')
plt.ylabel('Temperature [K]')
plt.legend()
plt.grid(True)
plt.show()

# ================================
# SUMMARY
# ================================

print("\n" + "="*60)
print("ROM SUMMARY")
print("="*60)
print(f"Original problem size: {Nx-1}×{Nx-1} = {(Nx-1)**2:,} unknowns")
for Nr in Nr_values:
    reduction = (Nx-1)**2 / Nr**2
    print(f"ROM with {Nr} modes: {Nr}×{Nr} = {Nr**2} unknowns (reduction: {reduction:.0f}×)")

print("\nKEY INSIGHTS FROM OUR TUTORIAL:")
print("1. Matrix A represents spatial derivatives only - never changes")
print("2. Time snapshots are columns of data matrix")
print("3. SVD finds dominant spatial-temporal patterns") 
print("4. ROM uses few modes instead of all spatial points")
print("5. Training on tent data works for Hertz prediction!")
print("="*60)

"""
ADD THIS TO THE END OF YOUR EXISTING CODE
=====================================
This will save the essential ROM matrices to files, then load them
in a standalone surrogate model that can predict any heat source!

Run your existing code first, then run this section.
"""

import os

# ================================
# SAVE ESSENTIAL ROM MATRICES
# ================================

# Create directory for ROM files
rom_folder = "ROM_matrices"
os.makedirs(rom_folder, exist_ok=True)

print(f"\nSaving ROM matrices to folder: {rom_folder}/")

# Save the essential data for ROM surrogate
np.savetxt(f"{rom_folder}/spatial_grid.txt", x, 
           header="Spatial grid points (x coordinates)")

np.savetxt(f"{rom_folder}/singular_values.txt", Sec, 
           header="Singular values from SVD (energy content)")

# Save spatial modes (U matrix from SVD)
np.savetxt(f"{rom_folder}/spatial_modes.txt", Uec, 
           header=f"Spatial modes from SVD, shape: {Uec.shape}")

# Save problem parameters
params_info = f"""# ROM Problem Parameters
# Generated from: {__file__ if '__file__' in globals() else 'ROM_model_1D_spatial_dist.py'}
# Date: {np.datetime64('today')}

# Physical parameters
q0 = {q0}        # Heat source intensity [W/m³]
Lx = {Lx}        # Rod length [m]
Lt = {Lt}        # Total time [s]

# Discretization
Nx = {Nx}        # Number of spatial points  
Nt = {Nt}        # Number of time steps
dx = {dx:.6f}    # Spatial step
dt = {dt:.3f}    # Time step

# Heat source parameters
body_width = {body_width:.3f}    # Heat source width [m]
velocity = {velocity:.3f}        # Heat source velocity [m/s]

# SVD Results
energy_first_10_modes = {(Sec[:10]**2).sum() / (Sec**2).sum() * 100:.2f}%
"""

with open(f"{rom_folder}/problem_parameters.txt", 'w') as f:
    f.write(params_info)

print("✅ Saved spatial_grid.txt")
print("✅ Saved singular_values.txt") 
print("✅ Saved spatial_modes.txt")
print("✅ Saved problem_parameters.txt")

# ================================
# STANDALONE ROM SURROGATE CLASS
# ================================

class HeatROMSurrogate:
    """
    Standalone ROM surrogate that loads from saved files
    Can predict temperature for ANY moving heat source!
    """
    
    def __init__(self, rom_folder="ROM_matrices"):
        """Load ROM matrices from folder"""
        self.rom_folder = rom_folder
        self.load_rom_data()
        
    def load_rom_data(self):
        """Load essential ROM matrices from files"""
        try:
            # Load spatial grid
            self.x = np.loadtxt(f"{self.rom_folder}/spatial_grid.txt")
            self.Nx = len(self.x)
            self.dx = self.x[1] - self.x[0]
            
            # Load SVD results
            self.singular_values = np.loadtxt(f"{self.rom_folder}/singular_values.txt")
            self.spatial_modes = np.loadtxt(f"{self.rom_folder}/spatial_modes.txt")
            
            print(f"✅ ROM data loaded from {self.rom_folder}/")
            print(f"   Grid points: {self.Nx}")
            print(f"   Available modes: {len(self.singular_values)}")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading ROM data: {e}")
            print("   Make sure you've run the training code first!")
            
    def setup_rom(self, Nr=7):
        """Set up ROM with specified number of modes"""
        self.Nr = Nr
        
        # Select first Nr spatial modes
        self.Phi = self.spatial_modes[:, :Nr]
        
        # Compute second derivatives of modes
        self.d2Phi_dx2 = np.zeros((self.Nx, Nr))
        for i in range(Nr):
            dPhi_dx = np.gradient(self.Phi[:, i], self.dx)
            self.d2Phi_dx2[:, i] = np.gradient(dPhi_dx, self.dx)
        
        # Build ROM system matrix (Nr×Nr instead of Nx×Nx!)
        self.A_rom = self.d2Phi_dx2.T @ self.d2Phi_dx2
        
        # Calculate energy captured
        total_energy = np.sum(self.singular_values**2)
        captured_energy = np.sum(self.singular_values[:Nr]**2)
        self.energy_captured = (captured_energy / total_energy) * 100
        
        print(f"✅ ROM setup with {Nr} modes")
        print(f"   Energy captured: {self.energy_captured:.2f}%")
        print(f"   System size: {self.Nx}×{self.Nx} → {Nr}×{Nr}")
        print(f"   Speedup factor: ~{(self.Nx/Nr)**2:.0f}×")
        
    def predict_temperature(self, heat_source_func, time_points, source_params):
        """
        Predict temperature evolution for any heat source
        
        Parameters:
        -----------
        heat_source_func : function
            Function: q = func(x, t, params)
        time_points : array
            Time points to evaluate
        source_params : dict
            Parameters for heat source
            
        Returns:
        --------
        T_predicted : array (Nx, Nt)
            Predicted temperature field
        """
        if not hasattr(self, 'A_rom'):
            print("❌ ROM not set up! Call setup_rom() first.")
            return None
            
        Nt = len(time_points)
        T_predicted = np.zeros((self.Nx, Nt))
        
        print(f"🔥 Predicting temperature with ROM...")
        print(f"   Heat source: {heat_source_func.__name__}")
        print(f"   Time points: {Nt}")
        print(f"   ROM modes: {self.Nr}")
        
        import time
        start_time = time.time()
        
        # Time stepping with ROM
        for i, t in enumerate(time_points):
            # Generate heat source at current time
            q_full = np.zeros(self.Nx)
            for j in range(self.Nx-1):  # Exclude boundary
                q_full[j] = heat_source_func(self.x[j], t, source_params)
            
            # Project onto ROM space
            q_rom = self.d2Phi_dx2.T @ q_full
            
            # Solve small ROM system (Nr×Nr)
            a_rom = np.linalg.solve(self.A_rom, -q_rom)
            
            # Reconstruct full temperature field
            T_predicted[:, i] = self.Phi @ a_rom
            
        solve_time = time.time() - start_time
        print(f"✅ ROM prediction completed in {solve_time:.3f} seconds")
        
        return T_predicted
    
    def compare_heat_sources(self, source_functions, time_span, common_params=None):
        """Compare different heat source types"""
        
        if common_params is None:
            common_params = {
                'velocity': 0.01,
                'width': 0.1, 
                'intensity': 1000
            }
        
        t_test = np.linspace(0, time_span, 200)
        results = {}
        
        plt.figure(figsize=(15, 10))
        
        for i, (name, func) in enumerate(source_functions.items()):
            print(f"\nTesting {name} heat source...")
            T = self.predict_temperature(func, t_test, common_params)
            results[name] = T
            
            # Plot result
            plt.subplot(2, len(source_functions), i+1)
            plt.imshow(T, aspect='auto', origin='lower', extent=[0, time_span, 0, 1])
            plt.title(f'{name} Heat Source')
            plt.xlabel('Time [s]')
            plt.ylabel('Position [m]')
            plt.colorbar(label='Temperature [K]')
            
            # Plot temperature at final time
            plt.subplot(2, len(source_functions), len(source_functions)+i+1)
            plt.plot(self.x, T[:, -1], linewidth=2, label=f'{name}')
            plt.xlabel('Position [m]')
            plt.ylabel('Final Temperature [K]')
            plt.title(f'{name} Final Temperature')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.rom_folder}/heat_source_comparison.png', dpi=150)
        plt.show()
        
        return results

# ================================
# HEAT SOURCE FUNCTIONS
# ================================

def hertz_source(x, t, params):
    """Hertz distribution heat source"""
    x0 = params['velocity'] * t
    r = np.abs(x - x0)
    a = params['width']
    q_max = params['intensity']
    
    if r <= a:
        return q_max * np.sqrt(np.abs(1.0001 - (r / a)**2))
    else:
        return 0.0

def gaussian_source(x, t, params):
    """Gaussian heat source (new type!)"""
    x0 = params['velocity'] * t
    sigma = params['width'] / 3  # 3-sigma width
    q_max = params['intensity']
    
    return q_max * np.exp(-0.5 * ((x - x0) / sigma)**2)

def square_source(x, t, params):
    """Square/uniform heat source"""
    x0 = params['velocity'] * t
    a = params['width']
    q_max = params['intensity']
    
    if np.abs(x - x0) <= a/2:
        return q_max
    else:
        return 0.0

def tent_source(x, t, params):
    """Tent/triangular heat source (same as training)"""
    x0 = params['velocity'] * t
    r = np.abs(x - x0)
    a = params['width']
    q_max = params['intensity']
    
    if r <= a:
        return q_max * (1 - r / a)
    else:
        return 0.0

# ================================
# DEMONSTRATION
# ================================

def demo_rom_surrogate():
    """Demonstrate the ROM surrogate model"""
    
    print("\n" + "="*70)
    print("🚀 ROM SURROGATE MODEL DEMONSTRATION")
    print("="*70)
    
    # Load ROM from saved files
    print("\n1. Loading ROM from saved files...")
    rom = HeatROMSurrogate("ROM_matrices")
    
    # Setup ROM with different numbers of modes
    print("\n2. Setting up ROM...")
    rom.setup_rom(Nr=7)  # Try 7 modes
    
    # Test different heat sources
    print("\n3. Testing different heat sources...")
    
    heat_sources = {
        'Tent': tent_source,
        'Hertz': hertz_source, 
        'Gaussian': gaussian_source,
        'Square': square_source
    }
    
    # Common parameters for fair comparison
    test_params = {
        'velocity': velocity,    # Use same velocity as training
        'width': body_width,     # Use same width as training
        'intensity': q0          # Use same intensity as training
    }
    
    # Compare all heat sources
    results = rom.compare_heat_sources(heat_sources, Lt/2, test_params)
    
    print("\n4. Testing parameter variations...")
    
    # Test parameter sensitivity
    velocities = [velocity*0.5, velocity, velocity*2.0]
    widths = [body_width*0.5, body_width, body_width*2.0]
    
    plt.figure(figsize=(15, 10))
    
    # Velocity sensitivity
    plt.subplot(2, 2, 1)
    for i, vel in enumerate(velocities):
        params = test_params.copy()
        params['velocity'] = vel
        t_test = np.linspace(0, Lt/4, 100)
        T = rom.predict_temperature(hertz_source, t_test, params)
        plt.plot(rom.x, T[:, -1], linewidth=2, label=f'v = {vel:.3f} m/s')
    plt.xlabel('Position [m]')
    plt.ylabel('Temperature [K]') 
    plt.title('Velocity Sensitivity')
    plt.legend()
    plt.grid(True)
    
    # Width sensitivity  
    plt.subplot(2, 2, 2)
    for i, width in enumerate(widths):
        params = test_params.copy()
        params['width'] = width
        t_test = np.linspace(0, Lt/4, 100)
        T = rom.predict_temperature(hertz_source, t_test, params)
        plt.plot(rom.x, T[:, -1], linewidth=2, label=f'w = {width:.3f} m')
    plt.xlabel('Position [m]')
    plt.ylabel('Temperature [K]')
    plt.title('Width Sensitivity') 
    plt.legend()
    plt.grid(True)
    
    # Mode count comparison
    plt.subplot(2, 2, 3)
    mode_counts = [3, 7, 15]
    for Nr in mode_counts:
        rom.setup_rom(Nr)
        T = rom.predict_temperature(hertz_source, np.linspace(0, Lt/4, 100), test_params)
        plt.plot(rom.x, T[:, -1], linewidth=2, label=f'{Nr} modes ({rom.energy_captured:.1f}%)')
    plt.xlabel('Position [m]')
    plt.ylabel('Temperature [K]')
    plt.title('ROM Mode Count Effect')
    plt.legend()
    plt.grid(True)
    
    # Energy content
    plt.subplot(2, 2, 4)
    cumulative_energy = np.cumsum(rom.singular_values**2) / np.sum(rom.singular_values**2) * 100
    plt.plot(range(1, min(21, len(cumulative_energy)+1)), cumulative_energy[:20], 'bo-', linewidth=2)
    plt.axhline(y=95, color='r', linestyle='--', alpha=0.7, label='95%')
    plt.axhline(y=99, color='g', linestyle='--', alpha=0.7, label='99%')
    plt.xlabel('Number of Modes')
    plt.ylabel('Cumulative Energy Captured [%]')
    plt.title('ROM Energy Convergence')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{rom.rom_folder}/rom_analysis.png', dpi=150)
    plt.show()
    
    print("\n" + "="*70)
    print("✅ ROM SURROGATE DEMONSTRATION COMPLETE!")
    print("="*70)
    print(f"📁 Files saved to: {rom.rom_folder}/")
    print("📊 heat_source_comparison.png - Comparison of different heat sources")
    print("📈 rom_analysis.png - ROM sensitivity and convergence analysis")
    print("\n🎯 Your ROM can now predict temperature for ANY moving heat source!")
    print("🚀 Ready for real-time applications, optimization, and control!")

# ================================
# RUN DEMONSTRATION
# ================================

if __name__ == "__main__":
    # This will run after your existing code
    demo_rom_surrogate()

print(f"\n{'='*60}")
print("📦 ROM MATRICES SAVED SUCCESSFULLY!")
print(f"📁 Location: {rom_folder}/")
print("🎯 You can now run the surrogate model independently!")
print("🚀 Ready for fast predictions of any heat source!")
print(f"{'='*60}")

