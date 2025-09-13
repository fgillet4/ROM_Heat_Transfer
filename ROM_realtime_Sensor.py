#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 12:11:55 2025

@author: francisbrain4
"""

#!/usr/bin/env python3
"""
ROM as Real-Time Virtual Sensor System
=====================================
Author: Francis Gillet
Date: September 12, 2025

This demonstrates how ROM can be used as a real-time virtual sensor for:
1. Process monitoring - estimate temperatures from process inputs
2. Anomaly detection - detect when process deviates from normal
3. Predictive control - predict future states for control decisions
4. Digital twin - real-time physics simulation

CONCEPT: The ROM becomes a "virtual thermometer" that estimates temperature
everywhere in the material based only on knowing the heat source parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import threading
import random

class ROMVirtualSensor:
    """
    ROM-based virtual sensor for real-time temperature estimation
    """
    
    def __init__(self, rom):
        """Initialize virtual sensor with trained ROM"""
        self.rom = rom
        self.is_monitoring = False
        self.sensor_data = deque(maxlen=1000)  # Store last 1000 readings
        self.prediction_times = deque(maxlen=100)
        
        # Sensor parameters
        self.update_rate = 10  # Hz
        self.dt = 1.0 / self.update_rate
        
        # Anomaly detection thresholds
        self.temp_threshold_max = 1200  # K
        self.temp_threshold_min = 50    # K
        self.uniformity_threshold = 100  # K standard deviation
        
        print("ROM Virtual Sensor initialized")
        print(f"Update rate: {self.update_rate} Hz")
        print(f"Prediction capability: ~{(self.rom.Nx/self.rom.Nr)**2:.0f}x faster than full simulation")
    
    def predict_current_state(self, current_params, current_time):
        """
        Predict current temperature state from process parameters
        
        This is the core "virtual sensor" functionality
        """
        start_time = time.time()
        
        # Create short time window for current prediction
        t_window = np.array([current_time])
        
        # ROM prediction
        try:
            T, _ = self.rom.predict(hertz_source, t_window, current_params, verbose=False)
            temp_field = T[:, 0]  # Current temperature distribution
            
            # Calculate key metrics
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
            anomalies.append(f"Overheating: {max_temp:.1f}K > {self.temp_threshold_max}K")
        
        if min_temp < self.temp_threshold_min:
            anomalies.append(f"Underheating: {min_temp:.1f}K < {self.temp_threshold_min}K")
        
        if uniformity > self.uniformity_threshold:
            anomalies.append(f"Poor uniformity: ±{uniformity:.1f}K > ±{self.uniformity_threshold}K")
        
        return anomalies
    
    def start_monitoring(self, process_scenario):
        """Start real-time monitoring"""
        print(f"\nStarting real-time monitoring...")
        print(f"Scenario: {process_scenario['name']}")
        
        self.is_monitoring = True
        self.sensor_data.clear()
        
        # Monitoring loop
        start_time = time.time()
        scenario_start = time.time()
        
        while self.is_monitoring:
            current_real_time = time.time() - start_time
            current_process_time = time.time() - scenario_start
            
            # Get current process parameters (simulated)
            current_params = self.simulate_process_inputs(process_scenario, current_process_time)
            
            if current_params is None:
                break  # End of scenario
            
            # Virtual sensor prediction
            sensor_reading = self.predict_current_state(current_params, current_process_time)
            
            if sensor_reading is not None:
                self.sensor_data.append(sensor_reading)
                self.prediction_times.append(sensor_reading['prediction_time'])
                
                # Print status
                print(f"Time: {current_process_time:6.1f}s | "
                      f"Max temp: {sensor_reading['max_temp']:6.1f}K | "
                      f"Uniformity: ±{sensor_reading['uniformity']:5.1f}K | "
                      f"Prediction: {sensor_reading['prediction_time']*1000:4.1f}ms", end="")
                
                if sensor_reading['anomaly']:
                    print(f" | ANOMALY: {sensor_reading['anomaly'][0]}")
                else:
                    print("")
            
            # Control update rate
            time.sleep(max(0, self.dt - (time.time() - start_time - current_real_time)))
        
        print("Monitoring stopped")
    
    def simulate_process_inputs(self, scenario, current_time):
        """Simulate realistic process input variations"""
        
        if current_time > scenario['duration']:
            return None  # End scenario
        
        base_params = scenario['base_params'].copy()
        
        # Add realistic variations based on scenario type
        if scenario['type'] == 'steady_welding':
            # Steady welding with small variations
            base_params['velocity'] += random.gauss(0, 0.001)  # ±1mm/s variation
            base_params['intensity'] += random.gauss(0, 50)    # ±50W/m³ variation
            
        elif scenario['type'] == 'speed_ramp':
            # Gradually increase welding speed
            ramp_factor = current_time / scenario['duration']
            base_params['velocity'] = scenario['base_params']['velocity'] * (1 + ramp_factor)
            
        elif scenario['type'] == 'power_disturbance':
            # Power disturbance at specific time
            if 15 < current_time < 25:
                base_params['intensity'] *= 0.7  # 30% power drop
                
        elif scenario['type'] == 'width_modulation':
            # Sinusoidal width modulation
            base_params['width'] = scenario['base_params']['width'] * (1 + 0.3 * np.sin(0.5 * current_time))
            
        return base_params
    
    def plot_monitoring_results(self):
        """Plot real-time monitoring results"""
        
        if not self.sensor_data:
            print("No monitoring data to plot")
            return
        
        # Extract data
        times = [reading['process_time'] for reading in self.sensor_data]
        max_temps = [reading['max_temp'] for reading in self.sensor_data]
        avg_temps = [reading['avg_temp'] for reading in self.sensor_data]
        uniformities = [reading['uniformity'] for reading in self.sensor_data]
        velocities = [reading['parameters']['velocity'] for reading in self.sensor_data]
        intensities = [reading['parameters']['intensity'] for reading in self.sensor_data]
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Temperature evolution
        axes[0, 0].plot(times, max_temps, 'r-', linewidth=2, label='Max Temperature')
        axes[0, 0].plot(times, avg_temps, 'b-', linewidth=2, label='Avg Temperature')
        axes[0, 0].axhline(self.temp_threshold_max, color='r', linestyle='--', alpha=0.5, label='Max Threshold')
        axes[0, 0].set_ylabel('Temperature [K]')
        axes[0, 0].set_title('Real-Time Temperature Monitoring')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Temperature uniformity
        axes[0, 1].plot(times, uniformities, 'g-', linewidth=2)
        axes[0, 1].axhline(self.uniformity_threshold, color='orange', linestyle='--', alpha=0.5, label='Uniformity Threshold')
        axes[0, 1].set_ylabel('Temperature Std Dev [K]')
        axes[0, 1].set_title('Temperature Uniformity')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Process parameters
        axes[1, 0].plot(times, velocities, 'm-', linewidth=2)
        axes[1, 0].set_ylabel('Velocity [m/s]')
        axes[1, 0].set_title('Process Input: Welding Speed')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(times, intensities, 'c-', linewidth=2)
        axes[1, 1].set_ylabel('Intensity [W/m³]')
        axes[1, 1].set_title('Process Input: Heat Intensity')
        axes[1, 1].grid(True)
        
        # Prediction performance
        pred_times_ms = [t * 1000 for t in self.prediction_times]
        axes[2, 0].hist(pred_times_ms, bins=20, alpha=0.7, color='orange')
        axes[2, 0].axvline(np.mean(pred_times_ms), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(pred_times_ms):.1f} ms')
        axes[2, 0].set_xlabel('Prediction Time [ms]')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('ROM Sensor Speed')
        axes[2, 0].legend()
        
        # Temperature field at final time
        if self.sensor_data:
            final_temp_field = self.sensor_data[-1]['temperature_field']
            axes[2, 1].plot(self.rom.x, final_temp_field, 'b-', linewidth=2)
            axes[2, 1].set_xlabel('Position [m]')
            axes[2, 1].set_ylabel('Temperature [K]')
            axes[2, 1].set_title('Final Temperature Distribution')
            axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('rom_sensor_monitoring.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_sensor_report(self):
        """Generate monitoring report"""
        
        if not self.sensor_data:
            return
        
        # Calculate statistics
        max_temps = [reading['max_temp'] for reading in self.sensor_data]
        uniformities = [reading['uniformity'] for reading in self.sensor_data]
        pred_times = [reading['prediction_time'] * 1000 for reading in self.sensor_data]
        
        anomaly_count = sum(1 for reading in self.sensor_data if reading['anomaly'])
        
        duration = self.sensor_data[-1]['process_time'] - self.sensor_data[0]['process_time']
        
        report = f"""
ROM VIRTUAL SENSOR MONITORING REPORT
===================================
Duration: {duration:.1f} seconds
Total readings: {len(self.sensor_data)}
Update rate: {len(self.sensor_data)/duration:.1f} Hz

TEMPERATURE STATISTICS:
- Max temperature range: {np.min(max_temps):.1f} - {np.max(max_temps):.1f} K
- Average temperature: {np.mean(max_temps):.1f} ± {np.std(max_temps):.1f} K
- Best uniformity: ±{np.min(uniformities):.1f} K
- Worst uniformity: ±{np.max(uniformities):.1f} K

ANOMALY DETECTION:
- Anomalies detected: {anomaly_count}
- Process uptime: {((len(self.sensor_data) - anomaly_count)/len(self.sensor_data)*100):.1f}%

ROM SENSOR PERFORMANCE:
- Average prediction time: {np.mean(pred_times):.2f} ms
- Max prediction time: {np.max(pred_times):.2f} ms
- Real-time capability: {np.mean(pred_times) < 1000/self.update_rate:.1f} (target: <{1000/self.update_rate:.1f} ms)

VIRTUAL SENSOR ACCURACY:
- Based on ROM with {self.rom.Nr} modes
- Energy captured: {self.rom.energy_captured:.1f}%
- System reduction: {(self.rom.Nx/self.rom.Nr)**2:.0f}x speedup vs full simulation
        """
        
        print(report)
        
        with open('rom_sensor_report.txt', 'w') as f:
            f.write(report)

class ROMPredictiveController:
    """
    ROM-based predictive controller for process optimization
    """
    
    def __init__(self, rom_sensor):
        self.sensor = rom_sensor
        self.rom = rom_sensor.rom
        self.control_active = False
        
        # Control parameters
        self.target_max_temp = 800   # K
        self.target_uniformity = 50  # K
        self.prediction_horizon = 10 # seconds
        
    def predict_future_state(self, current_params, prediction_time):
        """Predict future temperature state"""
        
        t_future = np.array([prediction_time])
        T, _ = self.rom.predict(hertz_source, t_future, current_params, verbose=False)
        
        return {
            'max_temp': np.max(T),
            'uniformity': np.std(T[:, 0])
        }
    
    def calculate_control_action(self, current_reading):
        """Calculate control adjustments"""
        
        current_params = current_reading['parameters'].copy()
        current_time = current_reading['process_time']
        
        # Predict future state with current parameters
        future_state = self.predict_future_state(current_params, 
                                                current_time + self.prediction_horizon)
        
        # Calculate errors
        temp_error = future_state['max_temp'] - self.target_max_temp
        uniformity_error = future_state['uniformity'] - self.target_uniformity
        
        # Simple control logic
        control_action = {}
        
        if temp_error > 50:  # Too hot
            control_action['velocity'] = min(current_params['velocity'] * 1.1, 0.03)  # Speed up
            control_action['intensity'] = max(current_params['intensity'] * 0.95, 500)  # Reduce power
        elif temp_error < -50:  # Too cold
            control_action['velocity'] = max(current_params['velocity'] * 0.9, 0.005)  # Slow down
            control_action['intensity'] = min(current_params['intensity'] * 1.05, 2000)  # Increase power
        
        if uniformity_error > 20:  # Poor uniformity
            control_action['width'] = min(current_params.get('width', 0.1) * 1.05, 0.2)  # Wider source
        
        return control_action

def demonstrate_virtual_sensor(rom):
    """Demonstrate ROM as virtual sensor"""
    
    print("ROM VIRTUAL SENSOR DEMONSTRATION")
    print("=" * 50)
    
    # Initialize virtual sensor
    sensor = ROMVirtualSensor(rom)
    
    # Define monitoring scenarios
    scenarios = [
        {
            'name': 'Steady Welding Process',
            'type': 'steady_welding',
            'duration': 30,
            'base_params': {'velocity': 0.015, 'width': 0.1, 'intensity': 1000}
        },
        {
            'name': 'Welding Speed Ramp',
            'type': 'speed_ramp', 
            'duration': 25,
            'base_params': {'velocity': 0.01, 'width': 0.1, 'intensity': 1200}
        },
        {
            'name': 'Power Disturbance',
            'type': 'power_disturbance',
            'duration': 35,
            'base_params': {'velocity': 0.012, 'width': 0.08, 'intensity': 1500}
        }
    ]
    
    # Run monitoring scenarios
    for i, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {i+1}: {scenario['name']} ---")
        
        # Start monitoring in separate thread for realism
        monitoring_thread = threading.Thread(target=sensor.start_monitoring, args=(scenario,))
        monitoring_thread.start()
        monitoring_thread.join()
        
        # Analyze results
        sensor.plot_monitoring_results()
        sensor.generate_sensor_report()
        
        if i < len(scenarios) - 1:
            input("Press Enter to continue to next scenario...")
    
    return sensor

def demonstrate_predictive_control(rom):
    """Demonstrate ROM for predictive control"""
    
    print("\nROM PREDICTIVE CONTROL DEMONSTRATION")
    print("=" * 50)
    
    sensor = ROMVirtualSensor(rom)
    controller = ROMPredictiveController(sensor)
    
    # Simulate control scenario
    print("Simulating predictive control for temperature regulation...")
    
    # Initial parameters (suboptimal)
    current_params = {'velocity': 0.008, 'width': 0.05, 'intensity': 1800}
    
    results = []
    
    for t in np.linspace(0, 20, 50):
        # Get current sensor reading
        reading = sensor.predict_current_state(current_params, t)
        
        if reading is None:
            continue
            
        # Calculate control action
        control_action = controller.calculate_control_action(reading)
        
        # Apply control (in real system, this would command actuators)
        for param, value in control_action.items():
            current_params[param] = value
        
        results.append({
            'time': t,
            'max_temp': reading['max_temp'],
            'uniformity': reading['uniformity'],
            'velocity': current_params['velocity'],
            'intensity': current_params['intensity']
        })
        
        print(f"t={t:5.1f}s: T_max={reading['max_temp']:6.1f}K, "
              f"uniform=±{reading['uniformity']:5.1f}K, "
              f"v={current_params['velocity']:.4f}m/s")
    
    # Plot control results
    times = [r['time'] for r in results]
    max_temps = [r['max_temp'] for r in results]
    uniformities = [r['uniformity'] for r in results]
    velocities = [r['velocity'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(times, max_temps, 'r-', linewidth=2)
    axes[0, 0].axhline(controller.target_max_temp, color='r', linestyle='--', alpha=0.5, label='Target')
    axes[0, 0].set_ylabel('Max Temperature [K]')
    axes[0, 0].set_title('Temperature Control')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(times, uniformities, 'g-', linewidth=2)
    axes[0, 1].axhline(controller.target_uniformity, color='g', linestyle='--', alpha=0.5, label='Target')
    axes[0, 1].set_ylabel('Temperature Uniformity [K]')
    axes[0, 1].set_title('Uniformity Control')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(times, velocities, 'm-', linewidth=2)
    axes[1, 0].set_ylabel('Velocity [m/s]')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_title('Control Action: Velocity')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(times, [r['intensity'] for r in results], 'c-', linewidth=2)
    axes[1, 1].set_ylabel('Intensity [W/m³]')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_title('Control Action: Intensity')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('rom_predictive_control.png', dpi=150, bbox_inches='tight')
    plt.show()

# Heat source function (same as before)
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

def main_sensor_demo(rom):
    """Main demonstration function"""
    
    print("ROM REAL-TIME SENSOR SYSTEM")
    print("=" * 60)
    print("This demonstrates ROM as:")
    print("1. Virtual temperature sensor")
    print("2. Anomaly detection system") 
    print("3. Predictive controller")
    print("4. Digital twin for process monitoring")
    
    # Virtual sensor demo
    sensor = demonstrate_virtual_sensor(rom)
    
    # Predictive control demo
    demonstrate_predictive_control(rom)
    
    print("\n" + "=" * 60)
    print("ROM SENSOR DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("Key capabilities demonstrated:")
    print("- Real-time temperature estimation from process inputs")
    print("- Millisecond prediction times suitable for control")
    print("- Anomaly detection and process monitoring")
    print("- Predictive control for temperature regulation")
    print("- Digital twin functionality for virtual sensing")
    
    return sensor

if __name__ == "__main__":
    print("Add this code to your ROM script to enable real-time sensor functionality!")
    # rom = HeatROMPredictor("ROM_matrices")
    # rom.setup_rom(Nr=7) 
    # main_sensor_demo(rom)