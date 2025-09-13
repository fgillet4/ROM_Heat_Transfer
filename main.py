#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive 1D Heat Equation ROM System with TUI
==================================================
Author: Francis Gillet
Date: September 12, 2025

Complete interactive system for:
- Training ROM models
- Running ROM predictions
- Parameter optimization
- Real-time simulation
- Testing and validation

Usage: python main.py
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Try to import dependencies, install if missing
try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    from scipy.optimize import minimize, differential_evolution
    import threading
    from collections import deque
    import random
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages: pip install scipy matplotlib numpy")
    sys.exit(1)

class HeatROMSystem:
    """Complete ROM system for 1D heat equation"""
    
    def __init__(self):
        self.rom_folder = "ROM_matrices"
        self.is_trained = False
        self.is_loaded = False
        self.is_setup = False
        
        # Create ROM folder if it doesn't exist
        os.makedirs(self.rom_folder, exist_ok=True)
        
        # Check if ROM matrices exist
        if self.check_rom_matrices_exist():
            self.load_rom_matrices()
    
    def check_rom_matrices_exist(self):
        """Check if ROM matrices already exist"""
        required_files = [
            'spatial_grid.txt',
            'singular_values.txt', 
            'spatial_modes.txt'
        ]
        
        for file in required_files:
            if not os.path.exists(f"{self.rom_folder}/{file}"):
                return False
        return True
    
    def train_rom_model(self, Nx=501, Nt=1000, q0=1e3, Lx=1, Lt=100):
        """Train ROM model from scratch"""
        
        print("="*60)
        print("TRAINING ROM MODEL")
        print("="*60)
        print(f"Grid: {Nx} spatial points, {Nt} time steps")
        
        # Create grids
        self.x = np.linspace(0, Lx, Nx)
        self.t = np.linspace(0, Lt, Nt)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]
        self.Nx = Nx
        self.Nt = Nt
        self.Lx = Lx
        self.Lt = Lt
        self.q0 = q0
        
        print(f"dx = {self.dx:.6f}, dt = {self.dt:.3f}")
        
        # Heat source parameters
        self.body_width = Lx / 10
        self.velocity = Lx / Lt
        
        print(f"Heat source width: {self.body_width:.3f} m")
        print(f"Heat source velocity: {self.velocity:.3f} m/s")
        
        # Build finite difference matrix
        print("\nBuilding finite difference matrix...")
        A = np.zeros((Nx-1, Nx-1))
        
        # Tridiagonal matrix for d²T/dx²
        np.fill_diagonal(A, -2)
        np.fill_diagonal(A[:, 1:], 1)
        np.fill_diagonal(A[1:, :], 1)
        
        # Boundary conditions
        A[0, 1] = 2  # dT/dx = 0 at x=0
        A = A / (self.dx**2)
        
        As = csr_matrix(A, shape=(Nx-1, Nx-1))
        
        # Generate training data using tent distribution
        print("\nGenerating training data...")
        qt = np.zeros(Nx-1)
        Xt = np.zeros((Nx, Nt))
        
        for i in range(Nt):
            for j in range(Nx-2):
                qt[j] = self.tent_distribution(self.x[j], self.body_width, q0, self.velocity, self.t[i])
            
            xt = spsolve(As, -qt)
            Xt[:-1, i] = xt
            Xt[-1, i] = 0
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{Nt} time steps")
        
        print(f"Training data generated: {Xt.shape}")
        
        # Perform SVD
        print("\nPerforming SVD analysis...")
        self.Uec, self.Sec, self.VTec = np.linalg.svd(Xt, full_matrices=False)
        
        print(f"SVD completed: U={self.Uec.shape}, S={self.Sec.shape}, VT={self.VTec.shape}")
        print(f"First 10 singular values: {self.Sec[:10]}")
        
        # Save ROM matrices
        self.save_rom_matrices()
        
        self.is_trained = True
        self.is_loaded = True
        
        print("✅ ROM training completed successfully!")
        
    def tent_distribution(self, x, a, q_max, velocity, time):
        """Tent (triangular) heat distribution"""
        x0 = velocity * time
        r = np.abs(x - x0)
        
        if r <= a:
            return q_max * (1 - r / a)
        else:
            return 0.0
    
    def save_rom_matrices(self):
        """Save ROM matrices to files"""
        
        print(f"\nSaving ROM matrices to {self.rom_folder}/")
        
        np.savetxt(f"{self.rom_folder}/spatial_grid.txt", self.x)
        np.savetxt(f"{self.rom_folder}/singular_values.txt", self.Sec)
        np.savetxt(f"{self.rom_folder}/spatial_modes.txt", self.Uec)
        
        # Save parameters
        params_info = f"""# ROM Problem Parameters
# Date: {datetime.now()}

# Physical parameters
q0 = {self.q0}
Lx = {self.Lx}
Lt = {self.Lt}

# Discretization
Nx = {self.Nx}
Nt = {self.Nt}
dx = {self.dx:.6f}
dt = {self.dt:.3f}

# Heat source parameters
body_width = {self.body_width:.3f}
velocity = {self.velocity:.3f}
"""
        
        with open(f"{self.rom_folder}/problem_parameters.txt", 'w') as f:
            f.write(params_info)
        
        print("✅ ROM matrices saved successfully!")
    
    def load_rom_matrices(self):
        """Load ROM matrices from files"""
        
        print(f"Loading ROM matrices from {self.rom_folder}/")
        
        try:
            self.x = np.loadtxt(f"{self.rom_folder}/spatial_grid.txt")
            self.Sec = np.loadtxt(f"{self.rom_folder}/singular_values.txt")
            self.Uec = np.loadtxt(f"{self.rom_folder}/spatial_modes.txt")
            
            self.Nx = len(self.x)
            self.dx = self.x[1] - self.x[0]
            self.Lx = self.x[-1] - self.x[0]
            
            self.is_loaded = True
            
            print("✅ ROM matrices loaded successfully!")
            print(f"  Spatial points: {self.Nx}")
            print(f"  Domain length: {self.Lx:.3f} m")
            print(f"  Available modes: {len(self.Sec)}")
            
        except Exception as e:
            print(f"❌ Error loading ROM matrices: {e}")
    
    def setup_rom(self, Nr=7):
        """Set up ROM with specified number of modes"""
        
        if not self.is_loaded:
            print("❌ ROM matrices not loaded!")
            return False
            
        self.Nr = Nr
        
        print(f"Setting up ROM with {Nr} modes...")
        
        # Select first Nr spatial modes
        self.Phi = self.Uec[:, :Nr]
        
        # Compute second derivatives
        self.d2Phi_dx2 = np.zeros((self.Nx, Nr))
        for i in range(Nr):
            dPhi_dx = np.gradient(self.Phi[:, i], self.dx)
            self.d2Phi_dx2[:, i] = np.gradient(dPhi_dx, self.dx)
        
        # Build ROM system matrix
        self.A_rom = self.d2Phi_dx2.T @ self.d2Phi_dx2
        
        # Calculate energy captured
        total_energy = np.sum(self.Sec**2)
        captured_energy = np.sum(self.Sec[:Nr]**2)
        self.energy_captured = (captured_energy / total_energy) * 100
        
        self.is_setup = True
        
        print(f"✅ ROM setup complete!")
        print(f"  Modes used: {Nr}")
        print(f"  Energy captured: {self.energy_captured:.2f}%")
        print(f"  System reduction: {self.Nx}×{self.Nx} → {Nr}×{Nr}")
        print(f"  Speedup factor: ~{(self.Nx/Nr)**2:.0f}×")
        
        return True
    
    def predict(self, heat_source_func, time_points, source_params, verbose=True):
        """Predict temperature evolution"""
        
        if not self.is_setup:
            print("❌ ROM not set up! Run setup_rom() first.")
            return None, 0
            
        Nt = len(time_points)
        T_predicted = np.zeros((self.Nx, Nt))
        
        if verbose:
            print(f"Predicting temperature evolution...")
            print(f"  Heat source: {heat_source_func.__name__}")
            print(f"  Time points: {Nt}")
            print(f"  Parameters: {source_params}")
        
        start_time = time.time()
        
        for i, t in enumerate(time_points):
            # Generate heat source
            q_full = np.zeros(self.Nx)
            for j in range(self.Nx-1):
                q_full[j] = heat_source_func(self.x[j], t, source_params)
            
            # ROM prediction
            q_rom = self.d2Phi_dx2.T @ q_full
            a_rom = np.linalg.solve(self.A_rom, -q_rom)
            T_predicted[:, i] = self.Phi @ a_rom
            
        prediction_time = time.time() - start_time
        
        if verbose:
            print(f"✅ Prediction completed in {prediction_time:.4f} seconds")
            print(f"  Max temperature: {np.max(T_predicted):.1f} K")
        
        return T_predicted, prediction_time

# Heat source functions
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
    """Gaussian heat distribution"""
    x0 = params['velocity'] * t
    sigma = params['width'] / 3
    q_max = params['intensity']
    
    return q_max * np.exp(-0.5 * ((x - x0) / sigma)**2)

def square_source(x, t, params):
    """Square heat distribution"""
    x0 = params['velocity'] * t
    a = params['width']
    q_max = params['intensity']
    
    if np.abs(x - x0) <= a/2:
        return q_max
    else:
        return 0.0

def tent_source(x, t, params):
    """Tent heat distribution"""
    x0 = params['velocity'] * t
    r = np.abs(x - x0)
    a = params['width']
    q_max = params['intensity']
    
    if r <= a:
        return q_max * (1 - r / a)
    else:
        return 0.0

class ROMOptimizer:
    """Parameter optimization for ROM"""
    
    def __init__(self, rom_system):
        self.rom = rom_system
        self.evaluation_count = 0
        
    def reset_counter(self):
        self.evaluation_count = 0
        
    def objective_function(self, params_array, objective_type='uniform_heating'):
        """Objective function for optimization"""
        
        self.evaluation_count += 1
        
        velocity, width, intensity = params_array
        
        params = {
            'velocity': velocity,
            'width': width,
            'intensity': intensity
        }
        
        process_time = self.rom.Lx / velocity
        t_eval = np.linspace(0, process_time, 100)
        
        try:
            T, _ = self.rom.predict(hertz_source, t_eval, params, verbose=False)
        except:
            return 1e6
        
        if objective_type == 'uniform_heating':
            return np.std(T[:, -1])
        elif objective_type == 'max_temp':
            target_max = 800
            return abs(np.max(T) - target_max)
        
        return np.std(T[:, -1])

class ROMVirtualSensor:
    """Real-time virtual sensor using ROM"""
    
    def __init__(self, rom_system):
        self.rom = rom_system
        self.is_monitoring = False
        self.sensor_data = deque(maxlen=1000)
        self.update_rate = 10  # Hz
        self.dt = 1.0 / self.update_rate
        
        # Thresholds
        self.temp_threshold_max = 1200
        self.temp_threshold_min = 50
        self.uniformity_threshold = 100
        
    def predict_current_state(self, current_params, current_time):
        """Predict current temperature state"""
        
        start_time = time.time()
        t_window = np.array([current_time])
        
        try:
            T, _ = self.rom.predict(hertz_source, t_window, current_params, verbose=False)
            temp_field = T[:, 0]
            
            max_temp = np.max(temp_field)
            min_temp = np.min(temp_field)
            avg_temp = np.mean(temp_field)
            uniformity = np.std(temp_field)
            
            prediction_time = time.time() - start_time
            
            sensor_reading = {
                'timestamp': time.time(),
                'process_time': current_time,
                'parameters': current_params.copy(),
                'temperature_field': temp_field,
                'max_temp': max_temp,
                'min_temp': min_temp,
                'avg_temp': avg_temp,
                'uniformity': uniformity,
                'prediction_time': prediction_time,
                'anomaly': self.detect_anomaly(max_temp, min_temp, uniformity)
            }
            
            return sensor_reading
            
        except Exception as e:
            print(f"Sensor prediction error: {e}")
            return None
    
    def detect_anomaly(self, max_temp, min_temp, uniformity):
        """Detect process anomalies"""
        anomalies = []
        
        if max_temp > self.temp_threshold_max:
            anomalies.append(f"Overheating: {max_temp:.1f}K")
        
        if min_temp < self.temp_threshold_min:
            anomalies.append(f"Underheating: {min_temp:.1f}K")
        
        if uniformity > self.uniformity_threshold:
            anomalies.append(f"Poor uniformity: ±{uniformity:.1f}K")
        
        return anomalies

class TUIMenu:
    """Text User Interface for ROM system"""
    
    def __init__(self):
        self.rom_system = HeatROMSystem()
        self.optimizer = None
        self.sensor = None
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_header(self):
        """Print main header"""
        print("="*70)
        print("    1D HEAT EQUATION ROM SYSTEM")
        print("    Advanced Reduced Order Modeling")
        print("    Author: Francis Gillet")
        print("="*70)
    
    def print_status(self):
        """Print system status"""
        print("\nSYSTEM STATUS:")
        print(f"  ROM Trained: {'✅' if self.rom_system.is_trained else '❌'}")
        print(f"  ROM Loaded: {'✅' if self.rom_system.is_loaded else '❌'}")
        print(f"  ROM Setup: {'✅' if self.rom_system.is_setup else '❌'}")
        
        if self.rom_system.is_setup:
            print(f"  Modes: {self.rom_system.Nr}")
            print(f"  Energy Captured: {self.rom_system.energy_captured:.1f}%")
    
    def main_menu(self):
        """Main menu loop"""
        
        while True:
            self.clear_screen()
            self.print_header()
            self.print_status()
            
            print("\nMAIN MENU:")
            print("1. Training & Setup")
            print("2. ROM Predictions")
            print("3. Parameter Optimization")
            print("4. Real-time Simulation")
            print("5. Testing & Validation")
            print("6. System Information")
            print("0. Exit")
            
            choice = input("\nEnter your choice (0-6): ").strip()
            
            if choice == '1':
                self.training_menu()
            elif choice == '2':
                self.prediction_menu()
            elif choice == '3':
                self.optimization_menu()
            elif choice == '4':
                self.realtime_menu()
            elif choice == '5':
                self.testing_menu()
            elif choice == '6':
                self.info_menu()
            elif choice == '0':
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice. Press Enter to continue...")
                input()
    
    def training_menu(self):
        """Training and setup menu"""
        
        while True:
            self.clear_screen()
            self.print_header()
            print("\nTRAINING & SETUP MENU:")
            print("1. Train new ROM model")
            print("2. Load existing ROM")
            print("3. Setup ROM with different modes")
            print("4. View training data plots")
            print("0. Back to main menu")
            
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == '1':
                print("\nTraining new ROM model...")
                print("This will generate training data and perform SVD analysis.")
                confirm = input("Continue? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    self.rom_system.train_rom_model()
                    self.rom_system.setup_rom(Nr=7)
                    print("\nTraining completed! Press Enter to continue...")
                    input()
                
            elif choice == '2':
                if self.rom_system.check_rom_matrices_exist():
                    self.rom_system.load_rom_matrices()
                    self.rom_system.setup_rom(Nr=7)
                    print("ROM loaded successfully! Press Enter to continue...")
                    input()
                else:
                    print("No existing ROM matrices found. Train a new model first.")
                    print("Press Enter to continue...")
                    input()
                
            elif choice == '3':
                if self.rom_system.is_loaded:
                    try:
                        modes = int(input("Enter number of modes (1-50): "))
                        if 1 <= modes <= 50:
                            self.rom_system.setup_rom(Nr=modes)
                            print("ROM setup updated! Press Enter to continue...")
                            input()
                        else:
                            print("Invalid number of modes.")
                            input()
                    except ValueError:
                        print("Invalid input.")
                        input()
                else:
                    print("No ROM data loaded. Load or train ROM first.")
                    input()
                
            elif choice == '4':
                if self.rom_system.is_loaded:
                    self.plot_training_analysis()
                else:
                    print("No ROM data loaded.")
                    input()
                
            elif choice == '0':
                break
    
    def prediction_menu(self):
        """Prediction menu"""
        
        if not self.rom_system.is_setup:
            print("ROM not setup. Please setup ROM first.")
            input("Press Enter to continue...")
            return
        
        while True:
            self.clear_screen()
            self.print_header()
            print("\nROM PREDICTIONS MENU:")
            print("1. Single heat source prediction")
            print("2. Compare heat source types")
            print("3. Parameter sensitivity study")
            print("4. Custom parameter sweep")
            print("0. Back to main menu")
            
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == '1':
                self.single_prediction()
            elif choice == '2':
                self.compare_heat_sources()
            elif choice == '3':
                self.parameter_sensitivity()
            elif choice == '4':
                self.custom_parameter_sweep()
            elif choice == '0':
                break
    
    def optimization_menu(self):
        """Optimization menu"""
        
        if not self.rom_system.is_setup:
            print("ROM not setup. Please setup ROM first.")
            input("Press Enter to continue...")
            return
        
        if not self.optimizer:
            self.optimizer = ROMOptimizer(self.rom_system)
        
        while True:
            self.clear_screen()
            self.print_header()
            print("\nPARAMETER OPTIMIZATION MENU:")
            print("1. Optimize for uniform heating")
            print("2. Optimize for target temperature")
            print("3. Parameter sweep analysis")
            print("4. Global optimization")
            print("0. Back to main menu")
            
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == '1':
                self.optimize_uniform_heating()
            elif choice == '2':
                self.optimize_target_temperature()
            elif choice == '3':
                self.parameter_sweep_analysis()
            elif choice == '4':
                self.global_optimization()
            elif choice == '0':
                break
    
    def realtime_menu(self):
        """Real-time simulation menu"""
        
        if not self.rom_system.is_setup:
            print("ROM not setup. Please setup ROM first.")
            input("Press Enter to continue...")
            return
        
        if not self.sensor:
            self.sensor = ROMVirtualSensor(self.rom_system)
        
        while True:
            self.clear_screen()
            self.print_header()
            print("\nREAL-TIME SIMULATION MENU:")
            print("1. Virtual sensor monitoring")
            print("2. Process anomaly detection")
            print("3. Live parameter tracking")
            print("4. Real-time control simulation")
            print("0. Back to main menu")
            
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == '1':
                self.virtual_sensor_demo()
            elif choice == '2':
                self.anomaly_detection_demo()
            elif choice == '3':
                self.parameter_tracking_demo()
            elif choice == '4':
                self.control_simulation_demo()
            elif choice == '0':
                break
    
    def testing_menu(self):
        """Testing and validation menu"""
        
        if not self.rom_system.is_setup:
            print("ROM not setup. Please setup ROM first.")
            input("Press Enter to continue...")
            return
        
        while True:
            self.clear_screen()
            self.print_header()
            print("\nTESTING & VALIDATION MENU:")
            print("1. ROM accuracy test")
            print("2. Speed benchmark")
            print("3. Mode convergence analysis")
            print("4. Cross-validation")
            print("0. Back to main menu")
            
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == '1':
                self.rom_accuracy_test()
            elif choice == '2':
                self.speed_benchmark()
            elif choice == '3':
                self.mode_convergence_analysis()
            elif choice == '4':
                self.cross_validation()
            elif choice == '0':
                break
    
    def info_menu(self):
        """System information menu"""
        
        self.clear_screen()
        self.print_header()
        print("\nSYSTEM INFORMATION:")
        
        if self.rom_system.is_loaded:
            print(f"Spatial points: {self.rom_system.Nx}")
            print(f"Domain length: {self.rom_system.Lx:.3f} m")
            print(f"Grid spacing: {self.rom_system.dx:.6f} m")
            print(f"Available modes: {len(self.rom_system.Sec)}")
            
            if self.rom_system.is_setup:
                print(f"Active modes: {self.rom_system.Nr}")
                print(f"Energy captured: {self.rom_system.energy_captured:.2f}%")
                print(f"System reduction: {(self.rom_system.Nx/self.rom_system.Nr)**2:.0f}x")
        else:
            print("No ROM data loaded.")
        
        print(f"\nROM folder: {self.rom_system.rom_folder}")
        print(f"Matrices exist: {self.rom_system.check_rom_matrices_exist()}")
        
        input("\nPress Enter to continue...")
    
    # Implementation of menu functions
    def single_prediction(self):
        """Single heat source prediction"""
        
        print("\nSINGLE HEAT SOURCE PREDICTION")
        print("Available heat sources:")
        print("1. Hertz (semicircular)")
        print("2. Gaussian")
        print("3. Square (uniform)")
        print("4. Tent (triangular)")
        
        try:
            source_choice = int(input("Choose heat source (1-4): "))
            velocity = float(input("Velocity [m/s] (default 0.01): ") or "0.01")
            width = float(input("Width [m] (default 0.1): ") or "0.1")
            intensity = float(input("Intensity [W/m³] (default 1000): ") or "1000")
            time_span = float(input("Time span [s] (default 50): ") or "50")
            
            source_funcs = {1: hertz_source, 2: gaussian_source, 3: square_source, 4: tent_source}
            source_names = {1: "Hertz", 2: "Gaussian", 3: "Square", 4: "Tent"}
            
            heat_source = source_funcs[source_choice]
            params = {'velocity': velocity, 'width': width, 'intensity': intensity}
            
            t_test = np.linspace(0, time_span, 200)
            T, pred_time = self.rom_system.predict(heat_source, t_test, params)
            
            # Plot results
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(T, aspect='auto', origin='lower', extent=[0, time_span, 0, self.rom_system.Lx])
            plt.title(f'{source_names[source_choice]} Heat Source Evolution')
            plt.xlabel('Time [s]')
            plt.ylabel('Position [m]')
            plt.colorbar(label='Temperature [K]')
            
            plt.subplot(1, 3, 2)
            plt.plot(self.rom_system.x, T[:, -1], linewidth=2)
            plt.title('Final Temperature Profile')
            plt.xlabel('Position [m]')
            plt.ylabel('Temperature [K]')
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            center_idx = self.rom_system.Nx // 2
            plt.plot(t_test, T[center_idx, :], linewidth=2)
            plt.title('Temperature at Center')
            plt.xlabel('Time [s]')
            plt.ylabel('Temperature [K]')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            print(f"\nPrediction completed in {pred_time:.4f} seconds")
            print(f"Max temperature: {np.max(T):.1f} K")
            print(f"Final temperature uniformity: ±{np.std(T[:, -1]):.1f} K")
            
        except (ValueError, KeyError):
            print("Invalid input.")
        
        input("Press Enter to continue...")
    
    def compare_heat_sources(self):
        """Compare different heat source types"""
        
        print("\nCOMPARING HEAT SOURCE TYPES")
        
        try:
            velocity = float(input("Velocity [m/s] (default 0.01): ") or "0.01")
            width = float(input("Width [m] (default 0.1): ") or "0.1")
            intensity = float(input("Intensity [W/m³] (default 1000): ") or "1000")
            time_span = float(input("Time span [s] (default 30): ") or "30")
            
            params = {'velocity': velocity, 'width': width, 'intensity': intensity}
            t_test = np.linspace(0, time_span, 150)
            
            heat_sources = {
                'Hertz': hertz_source,
                'Gaussian': gaussian_source,
                'Square': square_source,
                'Tent': tent_source
            }
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            
            for i, (name, func) in enumerate(heat_sources.items()):
                print(f"Simulating {name} heat source...")
                T, _ = self.rom_system.predict(func, t_test, params, verbose=False)
                
                # Temperature evolution
                axes[0, i].imshow(T, aspect='auto', origin='lower', 
                                extent=[0, time_span, 0, self.rom_system.Lx])
                axes[0, i].set_title(f'{name} Evolution')
                axes[0, i].set_xlabel('Time [s]')
                axes[0, i].set_ylabel('Position [m]')
                
                # Final profile
                axes[1, i].plot(self.rom_system.x, T[:, -1], linewidth=2)
                axes[1, i].set_title(f'{name} Final Profile')
                axes[1, i].set_xlabel('Position [m]')
                axes[1, i].set_ylabel('Temperature [K]')
                axes[1, i].grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ValueError:
            print("Invalid input.")
        
        input("Press Enter to continue...")
    
    def parameter_sensitivity(self):
        """Parameter sensitivity study"""
        
        print("\nPARAMETER SENSITIVITY STUDY")
        
        base_params = {'velocity': 0.01, 'width': 0.1, 'intensity': 1000}
        time_span = 25
        t_test = np.linspace(0, time_span, 100)
        
        # Test different parameter values
        velocities = [0.005, 0.01, 0.02, 0.03]
        widths = [0.05, 0.1, 0.15, 0.2]
        intensities = [500, 1000, 1500, 2000]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Velocity sensitivity
        print("Testing velocity sensitivity...")
        for vel in velocities:
            params = base_params.copy()
            params['velocity'] = vel
            T, _ = self.rom_system.predict(hertz_source, t_test, params, verbose=False)
            axes[0, 0].plot(self.rom_system.x, T[:, -1], linewidth=2, label=f'v={vel:.3f}')
        
        axes[0, 0].set_title('Velocity Sensitivity')
        axes[0, 0].set_xlabel('Position [m]')
        axes[0, 0].set_ylabel('Temperature [K]')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Width sensitivity
        print("Testing width sensitivity...")
        for width in widths:
            params = base_params.copy()
            params['width'] = width
            T, _ = self.rom_system.predict(hertz_source, t_test, params, verbose=False)
            axes[0, 1].plot(self.rom_system.x, T[:, -1], linewidth=2, label=f'w={width:.2f}')
        
        axes[0, 1].set_title('Width Sensitivity')
        axes[0, 1].set_xlabel('Position [m]')
        axes[0, 1].set_ylabel('Temperature [K]')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Intensity sensitivity
        print("Testing intensity sensitivity...")
        for intensity in intensities:
            params = base_params.copy()
            params['intensity'] = intensity
            T, _ = self.rom_system.predict(hertz_source, t_test, params, verbose=False)
            axes[0, 2].plot(self.rom_system.x, T[:, -1], linewidth=2, label=f'q={intensity}')
        
        axes[0, 2].set_title('Intensity Sensitivity')
        axes[0, 2].set_xlabel('Position [m]')
        axes[0, 2].set_ylabel('Temperature [K]')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Mode count effect
        print("Testing mode count effect...")
        original_Nr = self.rom_system.Nr
        mode_counts = [3, 7, 15, 25]
        
        for Nr in mode_counts:
            self.rom_system.setup_rom(Nr)
            T, _ = self.rom_system.predict(hertz_source, t_test, base_params, verbose=False)
            axes[1, 0].plot(self.rom_system.x, T[:, -1], linewidth=2, 
                           label=f'{Nr} modes ({self.rom_system.energy_captured:.1f}%)')
        
        axes[1, 0].set_title('Mode Count Effect')
        axes[1, 0].set_xlabel('Position [m]')
        axes[1, 0].set_ylabel('Temperature [K]')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Restore original mode count
        self.rom_system.setup_rom(original_Nr)
        
        # Energy convergence
        cumulative_energy = np.cumsum(self.rom_system.Sec**2) / np.sum(self.rom_system.Sec**2) * 100
        axes[1, 1].plot(range(1, min(21, len(cumulative_energy)+1)), cumulative_energy[:20], 'bo-')
        axes[1, 1].axhline(y=95, color='r', linestyle='--', alpha=0.7, label='95%')
        axes[1, 1].axhline(y=99, color='g', linestyle='--', alpha=0.7, label='99%')
        axes[1, 1].set_title('Energy Convergence')
        axes[1, 1].set_xlabel('Number of Modes')
        axes[1, 1].set_ylabel('Cumulative Energy [%]')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Prediction time analysis
        mode_range = range(1, 21)
        prediction_times = []
        
        for Nr in mode_range:
            self.rom_system.setup_rom(Nr)
            _, pred_time = self.rom_system.predict(hertz_source, t_test[:50], base_params, verbose=False)
            prediction_times.append(pred_time * 1000)  # Convert to ms
        
        axes[1, 2].plot(mode_range, prediction_times, 'ro-', linewidth=2)
        axes[1, 2].set_title('Prediction Time vs Modes')
        axes[1, 2].set_xlabel('Number of Modes')
        axes[1, 2].set_ylabel('Prediction Time [ms]')
        axes[1, 2].grid(True)
        
        # Restore original mode count
        self.rom_system.setup_rom(original_Nr)
        
        plt.tight_layout()
        plt.show()
        
        input("Press Enter to continue...")
    
    def optimize_uniform_heating(self):
        """Optimize for uniform heating"""
        
        print("\nOPTIMIZING FOR UNIFORM HEATING")
        print("This will find parameters that minimize temperature variation...")
        
        bounds = [(0.005, 0.03), (0.05, 0.2), (500, 2000)]  # velocity, width, intensity
        x0 = [0.015, 0.1, 1000]
        
        print("Running optimization...")
        self.optimizer.reset_counter()
        start_time = time.time()
        
        result = minimize(self.optimizer.objective_function, x0, 
                         args=('uniform_heating',), bounds=bounds, method='L-BFGS-B')
        
        opt_time = time.time() - start_time
        
        print(f"✅ Optimization completed in {opt_time:.2f} seconds")
        print(f"Function evaluations: {self.optimizer.evaluation_count}")
        print(f"Optimal parameters:")
        print(f"  Velocity: {result.x[0]:.4f} m/s")
        print(f"  Width: {result.x[1]:.3f} m")
        print(f"  Intensity: {result.x[2]:.0f} W/m³")
        print(f"Temperature uniformity: ±{result.fun:.1f} K")
        
        # Test optimal solution
        optimal_params = {
            'velocity': result.x[0],
            'width': result.x[1],
            'intensity': result.x[2]
        }
        
        process_time = self.rom_system.Lx / optimal_params['velocity']
        t_opt = np.linspace(0, process_time, 200)
        T_opt, _ = self.rom_system.predict(hertz_source, t_opt, optimal_params, verbose=False)
        
        # Compare with non-optimal
        suboptimal_params = {'velocity': 0.01, 'width': 0.05, 'intensity': 1500}
        t_sub = np.linspace(0, self.rom_system.Lx/0.01, 200)
        T_sub, _ = self.rom_system.predict(hertz_source, t_sub, suboptimal_params, verbose=False)
        
        # Plot comparison
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(T_opt, aspect='auto', origin='lower', extent=[0, process_time, 0, 1])
        plt.title('Optimal Temperature Evolution')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.plot(self.rom_system.x, T_opt[:, -1], 'b-', linewidth=2, 
                label=f'Optimal (±{np.std(T_opt[:, -1]):.1f} K)')
        plt.plot(self.rom_system.x, T_sub[:, -1], 'r--', linewidth=2, 
                label=f'Suboptimal (±{np.std(T_sub[:, -1]):.1f} K)')
        plt.xlabel('Position [m]')
        plt.ylabel('Final Temperature [K]')
        plt.title('Optimization Improvement')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(self.rom_system.x, T_opt[:, -1], 'b-', linewidth=2, label='Optimal')
        plt.fill_between(self.rom_system.x, 
                        T_opt[:, -1] - np.std(T_opt[:, -1]), 
                        T_opt[:, -1] + np.std(T_opt[:, -1]), 
                        alpha=0.3, label=f'±{np.std(T_opt[:, -1]):.1f} K')
        plt.xlabel('Position [m]')
        plt.ylabel('Temperature [K]')
        plt.title('Optimal Temperature Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        input("Press Enter to continue...")
    
    def virtual_sensor_demo(self):
        """Virtual sensor demonstration"""
        
        print("\nVIRTUAL SENSOR MONITORING DEMO")
        print("Simulating real-time process monitoring...")
        
        # Simulation parameters
        duration = 30  # seconds
        update_rate = 5  # Hz
        
        base_params = {'velocity': 0.015, 'width': 0.1, 'intensity': 1000}
        
        sensor_data = []
        times = np.linspace(0, duration, duration * update_rate)
        
        print(f"Running {duration}s simulation at {update_rate} Hz...")
        print("Time [s] | Max Temp [K] | Uniformity [K] | Prediction [ms] | Status")
        print("-" * 70)
        
        for t in times:
            # Add realistic variations
            current_params = base_params.copy()
            current_params['velocity'] += random.gauss(0, 0.001)
            current_params['intensity'] += random.gauss(0, 50)
            
            # ROM prediction
            reading = self.sensor.predict_current_state(current_params, t)
            
            if reading:
                sensor_data.append(reading)
                
                status = "NORMAL"
                if reading['anomaly']:
                    status = "ANOMALY"
                
                print(f"{t:6.1f}   | {reading['max_temp']:8.1f}   | "
                      f"{reading['uniformity']:8.1f}   | {reading['prediction_time']*1000:8.1f}   | {status}")
                
                time.sleep(0.1)  # Small delay for realism
        
        # Plot monitoring results
        if sensor_data:
            times = [r['process_time'] for r in sensor_data]
            max_temps = [r['max_temp'] for r in sensor_data]
            uniformities = [r['uniformity'] for r in sensor_data]
            pred_times = [r['prediction_time'] * 1000 for r in sensor_data]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            axes[0, 0].plot(times, max_temps, 'r-', linewidth=2)
            axes[0, 0].set_ylabel('Max Temperature [K]')
            axes[0, 0].set_title('Real-Time Temperature Monitoring')
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(times, uniformities, 'g-', linewidth=2)
            axes[0, 1].set_ylabel('Temperature Uniformity [K]')
            axes[0, 1].set_title('Temperature Uniformity')
            axes[0, 1].grid(True)
            
            axes[1, 0].hist(pred_times, bins=15, alpha=0.7, color='orange')
            axes[1, 0].axvline(np.mean(pred_times), color='r', linestyle='--', 
                              label=f'Mean: {np.mean(pred_times):.1f} ms')
            axes[1, 0].set_xlabel('Prediction Time [ms]')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('ROM Sensor Speed')
            axes[1, 0].legend()
            
            if sensor_data:
                final_temp_field = sensor_data[-1]['temperature_field']
                axes[1, 1].plot(self.rom_system.x, final_temp_field, 'b-', linewidth=2)
                axes[1, 1].set_xlabel('Position [m]')
                axes[1, 1].set_ylabel('Temperature [K]')
                axes[1, 1].set_title('Final Temperature Distribution')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.show()
            
            print(f"\nMonitoring completed!")
            print(f"Average prediction time: {np.mean(pred_times):.2f} ms")
            print(f"Max temperature range: {np.min(max_temps):.1f} - {np.max(max_temps):.1f} K")
            print(f"Best uniformity: ±{np.min(uniformities):.1f} K")
        
        input("Press Enter to continue...")
    
    def plot_training_analysis(self):
        """Plot training data analysis"""
        
        if not self.rom_system.is_loaded:
            print("No ROM data loaded.")
            return
        
        # Plot singular values and modes
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Singular values
        axes[0, 0].semilogy(self.rom_system.Sec[:20], 'bo-', linewidth=2)
        axes[0, 0].set_title('Singular Values')
        axes[0, 0].set_xlabel('Mode Number')
        axes[0, 0].set_ylabel('Singular Value')
        axes[0, 0].grid(True)
        
        # Cumulative energy
        cumulative_energy = np.cumsum(self.rom_system.Sec**2) / np.sum(self.rom_system.Sec**2) * 100
        axes[0, 1].plot(range(1, min(21, len(cumulative_energy)+1)), cumulative_energy[:20], 'ro-')
        axes[0, 1].axhline(y=95, color='g', linestyle='--', alpha=0.7, label='95%')
        axes[0, 1].axhline(y=99, color='orange', linestyle='--', alpha=0.7, label='99%')
        axes[0, 1].set_title('Cumulative Energy')
        axes[0, 1].set_xlabel('Number of Modes')
        axes[0, 1].set_ylabel('Energy Captured [%]')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # First few spatial modes
        modes_to_plot = min(6, self.rom_system.Uec.shape[1])
        for i in range(min(4, modes_to_plot)):
            row = i // 2
            col = (i % 2) + (0 if i < 2 else 1)
            if col < 3:
                mode_energy = self.rom_system.Sec[i]**2 / np.sum(self.rom_system.Sec**2) * 100
                axes[row, col].plot(self.rom_system.x, self.rom_system.Uec[:, i], linewidth=2)
                axes[row, col].set_title(f'Mode {i+1} ({mode_energy:.1f}% energy)')
                axes[row, col].set_xlabel('Position [m]')
                axes[row, col].set_ylabel('Amplitude')
                axes[row, col].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        input("Press Enter to continue...")
    
    def rom_accuracy_test(self):
        """Test ROM accuracy against reference"""
        
        print("\nROM ACCURACY TEST")
        print("Comparing ROM predictions with different mode counts...")
        
        # Test parameters
        test_params = {'velocity': 0.01, 'width': 0.1, 'intensity': 1000}
        time_span = 20
        t_test = np.linspace(0, time_span, 100)
        
        mode_counts = [3, 5, 7, 10, 15, 20]
        errors = []
        
        # Generate reference with highest mode count
        original_Nr = self.rom_system.Nr
        self.rom_system.setup_rom(50)  # High-fidelity reference
        T_ref, _ = self.rom_system.predict(hertz_source, t_test, test_params, verbose=False)
        
        print("Mode Count | Max Error [K] | RMS Error [K] | Relative Error [%]")
        print("-" * 60)
        
        for Nr in mode_counts:
            self.rom_system.setup_rom(Nr)
            T_rom, _ = self.rom_system.predict(hertz_source, t_test, test_params, verbose=False)
            
            error = np.abs(T_ref - T_rom)
            max_error = np.max(error)
            rms_error = np.sqrt(np.mean(error**2))
            rel_error = rms_error / np.max(T_ref) * 100
            
            errors.append({'modes': Nr, 'max_error': max_error, 'rms_error': rms_error, 'rel_error': rel_error})
            
            print(f"{Nr:9d}  | {max_error:10.2f}  | {rms_error:10.2f}  | {rel_error:12.2f}")
        
        # Plot error analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        mode_nums = [e['modes'] for e in errors]
        max_errors = [e['max_error'] for e in errors]
        rms_errors = [e['rms_error'] for e in errors]
        rel_errors = [e['rel_error'] for e in errors]
        
        axes[0, 0].semilogy(mode_nums, max_errors, 'bo-', linewidth=2, label='Max Error')
        axes[0, 0].semilogy(mode_nums, rms_errors, 'ro-', linewidth=2, label='RMS Error')
        axes[0, 0].set_xlabel('Number of Modes')
        axes[0, 0].set_ylabel('Error [K]')
        axes[0, 0].set_title('ROM Error Convergence')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(mode_nums, rel_errors, 'go-', linewidth=2)
        axes[0, 1].set_xlabel('Number of Modes')
        axes[0, 1].set_ylabel('Relative Error [%]')
        axes[0, 1].set_title('Relative Error vs Modes')
        axes[0, 1].grid(True)
        
        # Show specific comparison
        self.rom_system.setup_rom(7)
        T_7, _ = self.rom_system.predict(hertz_source, t_test, test_params, verbose=False)
        
        axes[1, 0].plot(self.rom_system.x, T_ref[:, -1], 'k-', linewidth=2, label='Reference (50 modes)')
        axes[1, 0].plot(self.rom_system.x, T_7[:, -1], 'r--', linewidth=2, label='ROM (7 modes)')
        axes[1, 0].set_xlabel('Position [m]')
        axes[1, 0].set_ylabel('Final Temperature [K]')
        axes[1, 0].set_title('ROM vs Reference')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        error_profile = np.abs(T_ref[:, -1] - T_7[:, -1])
        axes[1, 1].plot(self.rom_system.x, error_profile, 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Position [m]')
        axes[1, 1].set_ylabel('Absolute Error [K]')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Restore original mode count
        self.rom_system.setup_rom(original_Nr)
        
        input("Press Enter to continue...")
    
    def speed_benchmark(self):
        """Benchmark ROM prediction speed"""
        
        print("\nROM SPEED BENCHMARK")
        print("Testing prediction speed for different problem sizes...")
        
        test_params = {'velocity': 0.01, 'width': 0.1, 'intensity': 1000}
        
        # Test different time array sizes
        time_sizes = [10, 50, 100, 200, 500, 1000]
        mode_counts = [3, 7, 15, 25]
        
        results = []
        
        print("Time Points | Modes | Prediction Time [ms] | Speed [pts/ms]")
        print("-" * 55)
        
        for nt in time_sizes:
            t_test = np.linspace(0, 20, nt)
            
            for Nr in mode_counts:
                self.rom_system.setup_rom(Nr)
                
                # Multiple runs for better timing
                times = []
                for _ in range(3):
                    _, pred_time = self.rom_system.predict(hertz_source, t_test, test_params, verbose=False)
                    times.append(pred_time)
                
                avg_time = np.mean(times) * 1000  # Convert to ms
                speed = nt / avg_time  # points per ms
                
                results.append({
                    'time_points': nt,
                    'modes': Nr,
                    'time_ms': avg_time,
                    'speed': speed
                })
                
                print(f"{nt:10d}  | {Nr:4d}  | {avg_time:14.2f}  | {speed:11.2f}")
        
        # Plot benchmark results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time vs problem size
        for Nr in mode_counts:
            data = [r for r in results if r['modes'] == Nr]
            time_pts = [d['time_points'] for d in data]
            times = [d['time_ms'] for d in data]
            axes[0, 0].plot(time_pts, times, 'o-', linewidth=2, label=f'{Nr} modes')
        
        axes[0, 0].set_xlabel('Number of Time Points')
        axes[0, 0].set_ylabel('Prediction Time [ms]')
        axes[0, 0].set_title('Prediction Time vs Problem Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Speed vs modes
        for nt in [100, 500, 1000]:
            data = [r for r in results if r['time_points'] == nt]
            modes = [d['modes'] for d in data]
            speeds = [d['speed'] for d in data]
            axes[0, 1].plot(modes, speeds, 'o-', linewidth=2, label=f'{nt} time points')
        
        axes[0, 1].set_xlabel('Number of Modes')
        axes[0, 1].set_ylabel('Speed [time points/ms]')
        axes[0, 1].set_title('Speed vs ROM Size')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Speedup factor
        speedup_factors = [(self.rom_system.Nx / Nr)**2 for Nr in mode_counts]
        measured_times_1000 = [r['time_ms'] for r in results if r['time_points'] == 1000]
        
        if len(measured_times_1000) >= len(mode_counts):
            axes[1, 0].loglog(speedup_factors, measured_times_1000[:len(mode_counts)], 'ro-', linewidth=2)
            axes[1, 0].set_xlabel('Theoretical Speedup Factor')
            axes[1, 0].set_ylabel('Measured Time [ms]')
            axes[1, 0].set_title('Theoretical vs Measured Performance')
            axes[1, 0].grid(True)
        
        # Real-time capability analysis
        real_time_threshold = 100  # ms for 10 Hz updates
        
        for Nr in mode_counts:
            data = [r for r in results if r['modes'] == Nr]
            time_pts = [d['time_points'] for d in data]
            times = [d['time_ms'] for d in data]
            axes[1, 1].plot(time_pts, times, 'o-', linewidth=2, label=f'{Nr} modes')
        
        axes[1, 1].axhline(y=real_time_threshold, color='r', linestyle='--', 
                          label='Real-time threshold (10 Hz)')
        axes[1, 1].set_xlabel('Number of Time Points')
        axes[1, 1].set_ylabel('Prediction Time [ms]')
        axes[1, 1].set_title('Real-Time Capability')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nBenchmark Summary:")
        best_speed = max(results, key=lambda x: x['speed'])
        print(f"Best speed: {best_speed['speed']:.1f} time points/ms ({best_speed['modes']} modes)")
        
        real_time_capable = [r for r in results if r['time_ms'] < real_time_threshold]
        if real_time_capable:
            max_rt = max(real_time_capable, key=lambda x: x['time_points'])
            print(f"Max real-time problem size: {max_rt['time_points']} time points with {max_rt['modes']} modes")
        
        input("Press Enter to continue...")
    
    # Placeholder implementations for remaining methods
    def custom_parameter_sweep(self):
        print("Custom parameter sweep - Implementation in progress...")
        input("Press Enter to continue...")
    
    def optimize_target_temperature(self):
        print("Target temperature optimization - Implementation in progress...")
        input("Press Enter to continue...")
    
    def parameter_sweep_analysis(self):
        print("Parameter sweep analysis - Implementation in progress...")
        input("Press Enter to continue...")
    
    def global_optimization(self):
        print("Global optimization - Implementation in progress...")
        input("Press Enter to continue...")
    
    def anomaly_detection_demo(self):
        print("Anomaly detection demo - Implementation in progress...")
        input("Press Enter to continue...")
    
    def parameter_tracking_demo(self):
        print("Parameter tracking demo - Implementation in progress...")
        input("Press Enter to continue...")
    
    def control_simulation_demo(self):
        print("Control simulation demo - Implementation in progress...")
        input("Press Enter to continue...")
    
    def mode_convergence_analysis(self):
        print("Mode convergence analysis - Implementation in progress...")
        input("Press Enter to continue...")
    
    def cross_validation(self):
        print("Cross validation - Implementation in progress...")
        input("Press Enter to continue...")

def main():
    """Main entry point"""
    
    print("Starting 1D Heat Equation ROM System...")
    
    try:
        menu = TUIMenu()
        menu.main_menu()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please report this issue.")

if __name__ == "__main__":
    main()