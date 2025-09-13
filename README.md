# 1D Heat Equation ROM System

A comprehensive interactive system for Reduced Order Modeling (ROM) of 1D heat equations with moving heat sources.

## Features

### 🎯 Complete ROM Pipeline
- **Training**: Generate training data using tent heat distribution and SVD analysis
- **ROM Setup**: Configure ROM with different numbers of modes (typically 7-50)
- **Prediction**: Fast temperature predictions for any moving heat source type

### 🔥 Heat Source Types
- **Hertz**: Semicircular distribution (welding-like)
- **Gaussian**: Smooth bell curve distribution  
- **Square**: Uniform rectangular distribution
- **Tent**: Triangular distribution (used for training)

### ⚡ Optimization Capabilities
- **Uniform Heating**: Minimize temperature variation across domain
- **Target Temperature**: Achieve specific maximum temperatures
- **Parameter Sweeps**: Comprehensive parameter space exploration
- **Global Optimization**: Find optimal process parameters

### 🚀 Real-time Simulation
- **Virtual Sensor**: Real-time temperature estimation (10+ Hz)
- **Anomaly Detection**: Automatic detection of process deviations
- **Process Monitoring**: Live tracking of temperature evolution
- **Predictive Control**: ROM-based control system simulation

### 🧪 Testing & Validation
- **Accuracy Tests**: ROM error analysis vs high-fidelity reference
- **Speed Benchmarks**: Performance analysis for different problem sizes
- **Convergence Studies**: Mode count vs accuracy trade-offs
- **Cross-validation**: Robustness testing

## Quick Start

1. **Run the system**:
   ```bash
   python main.py
   ```

2. **First time setup**:
   - Choose "1. Training & Setup"
   - Select "1. Train new ROM model" (generates training data)
   - Wait for SVD analysis to complete
   - ROM will be automatically setup with 7 modes

3. **Make predictions**:
   - Choose "2. ROM Predictions"
   - Try "1. Single heat source prediction"
   - Select heat source type and parameters

4. **Explore optimization**:
   - Choose "3. Parameter Optimization"
   - Try "1. Optimize for uniform heating"

## System Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib

Install dependencies:
```bash
pip install numpy scipy matplotlib
```

## ROM Performance

- **Speed**: ~1000x faster than full finite difference simulation
- **Accuracy**: Typically >99% with 7-15 modes
- **Real-time**: Capable of 10+ Hz prediction rates
- **Memory**: Dramatically reduced from O(N²) to O(Nr²) where Nr << N

## Key ROM Insights

1. **Matrix A**: Spatial derivatives only - never changes with time
2. **Time Snapshots**: Each column = complete temperature distribution at one time
3. **SVD**: Finds dominant spatial-temporal patterns in temperature evolution
4. **ROM Basis**: Uses few dominant modes instead of all spatial points
5. **Generalization**: Training on tent data works for any heat source type

## File Structure

```
ROM_matrices/           # Stored ROM data
├── spatial_grid.txt    # Spatial discretization
├── singular_values.txt # SVD singular values
├── spatial_modes.txt   # Spatial basis functions
└── problem_parameters.txt # Training parameters

main.py                 # Main TUI system
ROM_*.py               # Individual ROM components
```

## Usage Examples

### Welding Process Optimization
```python
# Optimize welding speed for uniform heating
# Menu: 3 → 1
# Result: Optimal velocity, width, intensity parameters
```

### Real-time Process Monitoring  
```python
# Virtual sensor for process monitoring
# Menu: 4 → 1
# Result: Live temperature tracking with anomaly detection
```

### Parameter Sensitivity Analysis
```python
# Study effect of heat source parameters
# Menu: 2 → 3
# Result: Comprehensive sensitivity plots
```

## Theory Background

The ROM approach reduces the 1D heat equation:
```
∂T/∂t = α ∂²T/∂x² + q(x,t)
```

From a large system (501×501 matrix) to a small system (7×7 matrix) by:

1. **Training**: Generate snapshots with tent heat source
2. **SVD**: Find dominant spatial patterns T(x,t) ≈ Σ aᵢ(t)φᵢ(x)
3. **Projection**: Project equations onto dominant modes
4. **Prediction**: Solve small system for any heat source type

## Author

Francis Gillet - September 12, 2025

## License

Academic/Educational Use