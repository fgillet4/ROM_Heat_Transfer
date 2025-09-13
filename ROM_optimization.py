#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 12:04:19 2025

@author: francisbrain4
"""

#!/usr/bin/env python3
"""
ROM Parameter Modification and Optimization Guide
================================================
Author: Francis Gillet
Date: September 12, 2025

This script shows how to:
1. Modify ROM parameters easily
2. Set up optimization problems
3. Run parameter sweeps
4. Find optimal solutions

Add this to your standalone ROM script or run separately.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import time

# ================================
# PARAMETER MODIFICATION EXAMPLES
# ================================

def example_parameter_changes(rom):
    """Show how to easily change parameters"""
    
    print("PARAMETER MODIFICATION EXAMPLES")
    print("=" * 40)
    
    # Base time array
    t_test = np.linspace(0, 50, 200)
    
    # Example 1: Change welding speed
    print("\n1. Testing different welding speeds:")
    
    speeds = [0.005, 0.01, 0.02, 0.03]  # m/s
    
    plt.figure(figsize=(15, 10))
    
    for i, speed in enumerate(speeds):
        # Simply change the velocity parameter
        params = {
            'velocity': speed,      # CHANGE THIS
            'width': 0.1,
            'intensity': 1000
        }
        
        T, pred_time = rom.predict(hertz_source, t_test, params, verbose=False)
        
        print(f"   Speed {speed:.3f} m/s: Max temp = {np.max(T):.1f} K, Time = {pred_time:.4f}s")
        
        # Plot results
        plt.subplot(2, 2, i+1)
        plt.imshow(T, aspect='auto', origin='lower', extent=[0, 50, 0, 1])
        plt.title(f'Speed: {speed:.3f} m/s')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # Example 2: Change heat source width
    print("\n2. Testing different heat source widths:")
    
    widths = [0.05, 0.1, 0.2, 0.3]  # m
    
    plt.figure(figsize=(12, 8))
    
    for width in widths:
        params = {
            'velocity': 0.01,
            'width': width,         # CHANGE THIS
            'intensity': 1000
        }
        
        T, _ = rom.predict(gaussian_source, t_test, params, verbose=False)
        
        plt.plot(rom.x, T[:, -1], linewidth=2, label=f'Width = {width:.2f} m')
        
        print(f"   Width {width:.2f} m: Max temp = {np.max(T):.1f} K")
    
    plt.xlabel('Position [m]')
    plt.ylabel('Final Temperature [K]')
    plt.title('Effect of Heat Source Width')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Example 3: Change heat intensity
    print("\n3. Testing different heat intensities:")
    
    intensities = [500, 1000, 1500, 2000]  # W/m³
    
    plt.figure(figsize=(12, 8))
    
    for intensity in intensities:
        params = {
            'velocity': 0.01,
            'width': 0.1,
            'intensity': intensity  # CHANGE THIS
        }
        
        T, _ = rom.predict(tent_source, t_test, params, verbose=False)
        
        plt.plot(rom.x, T[:, -1], linewidth=2, label=f'Intensity = {intensity} W/m³')
        
        print(f"   Intensity {intensity} W/m³: Max temp = {np.max(T):.1f} K")
    
    plt.xlabel('Position [m]')
    plt.ylabel('Final Temperature [K]')
    plt.title('Effect of Heat Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

# ================================
# OPTIMIZATION SETUP
# ================================

class ROMOptimizer:
    """Class for optimizing ROM parameters"""
    
    def __init__(self, rom):
        self.rom = rom
        self.evaluation_count = 0
        
    def reset_counter(self):
        """Reset evaluation counter"""
        self.evaluation_count = 0
        
    def objective_function(self, params_array, target_profile=None, objective_type='uniform_heating'):
        """
        Objective function for optimization
        
        Parameters:
        -----------
        params_array : array
            [velocity, width, intensity] to optimize
        target_profile : array, optional
            Target temperature profile
        objective_type : str
            Type of objective ('uniform_heating', 'max_temp', 'target_matching')
        """
        
        self.evaluation_count += 1
        
        # Unpack parameters
        velocity, width, intensity = params_array
        
        # Create parameter dictionary
        params = {
            'velocity': velocity,
            'width': width,
            'intensity': intensity
        }
        
        # Predict temperature with ROM
        process_time = self.rom.Lx / velocity  # Time to cross domain
        t_eval = np.linspace(0, process_time, 100)
        
        try:
            T, _ = self.rom.predict(hertz_source, t_eval, params, verbose=False)
        except:
            return 1e6  # Return large penalty for failed predictions
        
        # Calculate objective based on type
        if objective_type == 'uniform_heating':
            # Minimize temperature variation
            final_temps = T[:, -1]
            objective = np.std(final_temps)
            
        elif objective_type == 'max_temp':
            # Achieve specific maximum temperature
            target_max = 800  # K
            actual_max = np.max(T)
            objective = abs(actual_max - target_max)
            
        elif objective_type == 'target_matching':
            # Match a target temperature profile
            if target_profile is None:
                target_profile = np.ones(self.rom.Nx) * 500  # Default target
            
            final_temps = T[:, -1]
            objective = np.sum((final_temps - target_profile)**2)
            
        elif objective_type == 'energy_efficiency':
            # Minimize energy while achieving minimum temperature
            min_required_temp = 400  # K
            actual_min = np.min(T)
            
            if actual_min < min_required_temp:
                objective = 1e6 + (min_required_temp - actual_min) * 1000  # Penalty
            else:
                objective = intensity  # Minimize energy input
                
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")
        
        return objective

def run_parameter_sweep(rom):
    """Run comprehensive parameter sweep"""
    
    print("PARAMETER SWEEP ANALYSIS")
    print("=" * 30)
    
    # Define parameter ranges
    velocities = np.linspace(0.005, 0.03, 8)
    widths = np.linspace(0.05, 0.2, 6)
    intensities = np.linspace(500, 2000, 6)
    
    # Storage for results
    results = {
        'velocity': [],
        'width': [],
        'intensity': [],
        'max_temp': [],
        'temp_uniformity': [],
        'prediction_time': []
    }
    
    total_combinations = len(velocities) * len(widths) * len(intensities)
    print(f"Testing {total_combinations} parameter combinations...")
    
    start_time = time.time()
    count = 0
    
    # Test all combinations
    for vel in velocities:
        for width in widths:
            for intensity in intensities:
                count += 1
                
                params = {
                    'velocity': vel,
                    'width': width,
                    'intensity': intensity
                }
                
                # ROM prediction
                process_time = rom.Lx / vel
                t_test = np.linspace(0, process_time, 100)
                
                T, pred_time = rom.predict(hertz_source, t_test, params, verbose=False)
                
                # Store results
                results['velocity'].append(vel)
                results['width'].append(width)
                results['intensity'].append(intensity)
                results['max_temp'].append(np.max(T))
                results['temp_uniformity'].append(np.std(T[:, -1]))
                results['prediction_time'].append(pred_time)
                
                if count % 50 == 0:
                    print(f"   Completed {count}/{total_combinations} combinations...")
    
    total_time = time.time() - start_time
    print(f"Parameter sweep completed in {total_time:.2f} seconds!")
    print(f"Average time per prediction: {total_time/total_combinations*1000:.2f} ms")
    
    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])
    
    # Analysis
    print("\nParameter Sweep Results:")
    print(f"Temperature range: {np.min(results['max_temp']):.1f} - {np.max(results['max_temp']):.1f} K")
    print(f"Best uniformity: ±{np.min(results['temp_uniformity']):.1f} K")
    print(f"Worst uniformity: ±{np.max(results['temp_uniformity']):.1f} K")
    
    # Find optimal parameters for uniform heating
    best_idx = np.argmin(results['temp_uniformity'])
    print(f"\nBest parameters for uniform heating:")
    print(f"   Velocity: {results['velocity'][best_idx]:.4f} m/s")
    print(f"   Width: {results['width'][best_idx]:.3f} m")
    print(f"   Intensity: {results['intensity'][best_idx]:.0f} W/m³")
    print(f"   Uniformity: ±{results['temp_uniformity'][best_idx]:.1f} K")
    
    # Visualization
    plot_parameter_sweep_results(results)
    
    return results

def plot_parameter_sweep_results(results):
    """Plot parameter sweep results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Max temperature vs velocity
    axes[0, 0].scatter(results['velocity'], results['max_temp'], 
                      c=results['intensity'], alpha=0.6, s=20)
    axes[0, 0].set_xlabel('Velocity [m/s]')
    axes[0, 0].set_ylabel('Max Temperature [K]')
    axes[0, 0].set_title('Max Temperature vs Velocity')
    plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label='Intensity [W/m³]')
    
    # Temperature uniformity vs width
    axes[0, 1].scatter(results['width'], results['temp_uniformity'], 
                      c=results['velocity'], alpha=0.6, s=20)
    axes[0, 1].set_xlabel('Width [m]')
    axes[0, 1].set_ylabel('Temperature Uniformity [K]')
    axes[0, 1].set_title('Uniformity vs Width')
    plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='Velocity [m/s]')
    
    # 3D scatter: velocity vs width vs uniformity
    sc = axes[0, 2].scatter(results['velocity'], results['width'], 
                           c=results['temp_uniformity'], s=30, alpha=0.7)
    axes[0, 2].set_xlabel('Velocity [m/s]')
    axes[0, 2].set_ylabel('Width [m]')
    axes[0, 2].set_title('Parameter Space (colored by uniformity)')
    plt.colorbar(sc, ax=axes[0, 2], label='Uniformity [K]')
    
    # Max temp vs intensity
    axes[1, 0].scatter(results['intensity'], results['max_temp'], 
                      c=results['width'], alpha=0.6, s=20)
    axes[1, 0].set_xlabel('Intensity [W/m³]')
    axes[1, 0].set_ylabel('Max Temperature [K]')
    axes[1, 0].set_title('Max Temperature vs Intensity')
    plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='Width [m]')
    
    # Pareto front: uniformity vs max temperature
    axes[1, 1].scatter(results['temp_uniformity'], results['max_temp'], 
                      c=results['velocity'], alpha=0.6, s=20)
    axes[1, 1].set_xlabel('Temperature Uniformity [K]')
    axes[1, 1].set_ylabel('Max Temperature [K]')
    axes[1, 1].set_title('Pareto Front: Uniformity vs Max Temp')
    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Velocity [m/s]')
    
    # Prediction time histogram
    axes[1, 2].hist(results['prediction_time'] * 1000, bins=20, alpha=0.7)
    axes[1, 2].set_xlabel('Prediction Time [ms]')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('ROM Prediction Speed')
    axes[1, 2].axvline(np.mean(results['prediction_time']) * 1000, 
                      color='r', linestyle='--', label=f'Mean: {np.mean(results["prediction_time"])*1000:.2f} ms')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('parameter_sweep_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_optimization_examples(rom):
    """Run different optimization examples"""
    
    print("OPTIMIZATION EXAMPLES")
    print("=" * 25)
    
    optimizer = ROMOptimizer(rom)
    
    # Optimization Example 1: Uniform Heating
    print("\n1. Optimizing for uniform heating...")
    
    optimizer.reset_counter()
    
    # Parameter bounds: [velocity, width, intensity]
    bounds = [(0.005, 0.03), (0.05, 0.2), (500, 2000)]
    
    # Initial guess
    x0 = [0.015, 0.1, 1000]
    
    # Run optimization
    start_time = time.time()
    result = minimize(optimizer.objective_function, x0, 
                     args=(None, 'uniform_heating'),
                     bounds=bounds, method='L-BFGS-B')
    opt_time = time.time() - start_time
    
    print(f"   Optimization completed in {opt_time:.2f} seconds")
    print(f"   Function evaluations: {optimizer.evaluation_count}")
    print(f"   Optimal parameters:")
    print(f"     Velocity: {result.x[0]:.4f} m/s")
    print(f"     Width: {result.x[1]:.3f} m")
    print(f"     Intensity: {result.x[2]:.0f} W/m³")
    print(f"   Objective value: ±{result.fun:.1f} K uniformity")
    
    # Test optimal solution
    optimal_params = {
        'velocity': result.x[0],
        'width': result.x[1],
        'intensity': result.x[2]
    }
    
    process_time = rom.Lx / optimal_params['velocity']
    t_opt = np.linspace(0, process_time, 200)
    T_opt, _ = rom.predict(hertz_source, t_opt, optimal_params, verbose=False)
    
    # Plot optimal solution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(T_opt, aspect='auto', origin='lower', extent=[0, process_time, 0, 1])
    plt.title('Optimal Temperature Evolution')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.plot(rom.x, T_opt[:, -1], 'b-', linewidth=2, label='Optimal')
    plt.xlabel('Position [m]')
    plt.ylabel('Final Temperature [K]')
    plt.title('Optimal Temperature Profile')
    plt.grid(True)
    plt.legend()
    
    # Compare with non-optimal
    suboptimal_params = {'velocity': 0.01, 'width': 0.05, 'intensity': 1500}
    t_sub = np.linspace(0, rom.Lx/0.01, 200)
    T_sub, _ = rom.predict(hertz_source, t_sub, suboptimal_params, verbose=False)
    
    plt.subplot(1, 3, 3)
    plt.plot(rom.x, T_opt[:, -1], 'b-', linewidth=2, label=f'Optimal (±{np.std(T_opt[:, -1]):.1f} K)')
    plt.plot(rom.x, T_sub[:, -1], 'r--', linewidth=2, label=f'Suboptimal (±{np.std(T_sub[:, -1]):.1f} K)')
    plt.xlabel('Position [m]')
    plt.ylabel('Final Temperature [K]')
    plt.title('Optimization Improvement')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Optimization Example 2: Global optimization with differential evolution
    print("\n2. Global optimization with differential evolution...")
    
    optimizer.reset_counter()
    start_time = time.time()
    
    result_global = differential_evolution(optimizer.objective_function, bounds,
                                         args=(None, 'uniform_heating'),
                                         maxiter=50, popsize=15)
    
    global_time = time.time() - start_time
    print(f"   Global optimization completed in {global_time:.2f} seconds")
    print(f"   Function evaluations: {optimizer.evaluation_count}")
    print(f"   Global optimal parameters:")
    print(f"     Velocity: {result_global.x[0]:.4f} m/s")
    print(f"     Width: {result_global.x[1]:.3f} m") 
    print(f"     Intensity: {result_global.x[2]:.0f} W/m³")
    print(f"   Global objective value: ±{result_global.fun:.1f} K uniformity")

# ================================
# EASY PARAMETER MODIFICATION FUNCTIONS
# ================================

def quick_parameter_test(rom, param_name, param_values, base_params=None):
    """Quickly test different values of a single parameter"""
    
    if base_params is None:
        base_params = {'velocity': 0.01, 'width': 0.1, 'intensity': 1000}
    
    print(f"Testing different values of {param_name}:")
    
    results = []
    t_test = np.linspace(0, 50, 100)
    
    for value in param_values:
        # Copy base parameters and modify the target parameter
        params = base_params.copy()
        params[param_name] = value
        
        # Run ROM prediction
        T, pred_time = rom.predict(hertz_source, t_test, params, verbose=False)
        
        max_temp = np.max(T)
        uniformity = np.std(T[:, -1])
        
        results.append({
            'value': value,
            'max_temp': max_temp,
            'uniformity': uniformity,
            'pred_time': pred_time
        })
        
        print(f"   {param_name} = {value}: Max temp = {max_temp:.1f} K, Uniformity = ±{uniformity:.1f} K")
    
    return results

def create_design_of_experiments(rom, n_samples=100):
    """Create a design of experiments for parameter exploration"""
    
    print(f"Creating design of experiments with {n_samples} samples...")
    
    # Latin Hypercube sampling for better space coverage
    from scipy.stats import qmc
    
    sampler = qmc.LatinHypercube(d=3)  # 3 parameters
    samples = sampler.random(n=n_samples)
    
    # Scale to parameter ranges
    param_ranges = {
        'velocity': (0.005, 0.03),
        'width': (0.05, 0.2), 
        'intensity': (500, 2000)
    }
    
    scaled_samples = qmc.scale(samples, 
                              [param_ranges['velocity'][0], param_ranges['width'][0], param_ranges['intensity'][0]],
                              [param_ranges['velocity'][1], param_ranges['width'][1], param_ranges['intensity'][1]])
    
    # Run experiments
    doe_results = []
    
    for i, (vel, width, intensity) in enumerate(scaled_samples):
        params = {
            'velocity': vel,
            'width': width,
            'intensity': intensity
        }
        
        process_time = rom.Lx / vel
        t_test = np.linspace(0, process_time, 50)  # Fewer time points for speed
        
        T, pred_time = rom.predict(hertz_source, t_test, params, verbose=False)
        
        doe_results.append({
            'velocity': vel,
            'width': width,
            'intensity': intensity,
            'max_temp': np.max(T),
            'uniformity': np.std(T[:, -1]),
            'energy_input': intensity * width,  # Simplified energy metric
            'pred_time': pred_time
        })
        
        if (i + 1) % 20 == 0:
            print(f"   Completed {i + 1}/{n_samples} experiments...")
    
    print("Design of experiments completed!")
    return doe_results

# Include the heat source functions from previous code
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

# ================================
# MAIN DEMONSTRATION FUNCTION
# ================================

def demonstrate_parameter_optimization(rom):
    """Main function to demonstrate all parameter modification and optimization"""
    
    print("ROM PARAMETER MODIFICATION AND OPTIMIZATION DEMO")
    print("=" * 60)
    
    # 1. Basic parameter changes
    example_parameter_changes(rom)
    
    # 2. Quick parameter testing
    print("\nQuick parameter testing:")
    velocity_results = quick_parameter_test(rom, 'velocity', [0.005, 0.01, 0.02, 0.03])
    
    # 3. Parameter sweep
    sweep_results = run_parameter_sweep(rom)
    
    # 4. Optimization examples
    run_optimization_examples(rom)
    
    # 5. Design of experiments
    doe_results = create_design_of_experiments(rom, n_samples=50)
    
    print("\n" + "=" * 60)
    print("PARAMETER OPTIMIZATION DEMO COMPLETE!")
    print("=" * 60)
    print("You now know how to:")
    print("1. Modify any ROM parameter easily")
    print("2. Run parameter sweeps and sensitivity studies") 
    print("3. Set up and solve optimization problems")
    print("4. Use advanced sampling techniques")
    print("5. Analyze results and find optimal solutions")

if __name__ == "__main__":
    # This assumes you have rom loaded from the main script
    # rom = HeatROMPredictor("ROM_matrices")
    # rom.setup_rom(Nr=7)
    # demonstrate_parameter_optimization(rom)
    
    print("Add this code to your standalone ROM script to enable optimization!")