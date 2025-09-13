import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import svd

class Heat1D_ROM:
    def __init__(self, L=1000e-3, nx=1000, t_end=1000):
        """
        1D Heat Equation Solver with Reduced Order Modeling
        
        Parameters:
        L: domain length [m]
        nx: number of grid points
        t_end: simulation time [s]
        """
        self.L = L
        self.nx = nx
        self.t_end = t_end
        self.dx = L / (nx - 1)
        self.x = np.linspace(0, L, nx)
        
        # Problem parameters
        self.Ex_k_max = 1000  # Maximum Ex/k [K/m²]
        self.tent_width = L / 10
        
        # Setup finite difference matrix for d²T/dx²
        self._setup_diff_matrix()
        
    def _setup_diff_matrix(self):
        """Setup finite difference matrix for second derivative with BC"""
        # Create matrix like your friend's approach
        # We'll work with (nx-1) nodes since the last node is fixed at T=0
        n = self.nx - 1  # Number of unknowns (excluding right boundary)
        
        # Initialize matrix
        A = np.zeros((n, n))
        np.fill_diagonal(A, -2)
        np.fill_diagonal(A[:, 1:], 1)
        np.fill_diagonal(A[1:, :], 1)
        
        # Left BC: dT/dx = 0 (Neumann) using ghost point method
        # T_(-1) = T_(1), so first equation becomes: -2*T_0 + 2*T_1 = q_0
        A[0, 0] = -2
        A[0, 1] = 2
        
        # Scale by 1/dx^2
        A = A / (self.dx**2)
        
        # Convert to sparse matrix
        self.A = csr_matrix(A, shape=(n, n))
        
    def tent_distribution(self, t):
        """Moving tent heat source distribution"""
        # Position of tent center
        x_center = (t / self.t_end) * self.L
        
        # Tent function
        source = np.zeros_like(self.x)
        for i, xi in enumerate(self.x):
            dist = abs(xi - x_center)
            if dist <= self.tent_width / 2:
                source[i] = self.Ex_k_max * (1 - 2 * dist / self.tent_width)
        
        return source
    
    def hertzian_distribution(self, t):
        """Moving Hertzian (parabolic) heat source distribution"""
        # Position of center
        x_center = (t / self.t_end) * self.L
        
        # Hertzian contact distribution (parabolic)
        source = np.zeros_like(self.x)
        for i, xi in enumerate(self.x):
            dist = abs(xi - x_center)
            if dist <= self.tent_width / 2:
                source[i] = self.Ex_k_max * np.sqrt(1 - (2 * dist / self.tent_width)**2)
        
        return source
    
    def solve_high_resolution(self, source_func, time_steps=100):
        """
        Solve the stationary heat equation for different source positions
        Returns temperature snapshots matrix
        """
        print(f"Solving high-resolution problem with {self.nx} grid points...")
        
        time_points = np.linspace(0, self.t_end, time_steps)
        snapshots = np.zeros((self.nx, time_steps))
        
        for i, t in enumerate(time_points):
            # Get source term (only for internal nodes, excluding right boundary)
            source_internal = source_func(t)[:-1]  # Remove last node
            
            # Solve AT = -Ex/k for internal nodes
            T_internal = spsolve(self.A, -source_internal)
            
            # Reconstruct full solution with boundary condition
            T_full = np.zeros(self.nx)
            T_full[:-1] = T_internal
            T_full[-1] = 0  # Right boundary condition T(L) = 0
            
            snapshots[:, i] = T_full
            
            if i % (time_steps // 10) == 0:
                print(f"  Progress: {100*i//time_steps}%")
        
        self.time_points = time_points
        return snapshots
    
    def perform_svd(self, snapshots):
        """Perform SVD analysis on snapshot matrix"""
        print("Performing SVD analysis...")
        
        # Direct SVD without mean centering (like your friend's approach)
        # SVD: X = U * S * V^T
        U, s, Vt = np.linalg.svd(snapshots, full_matrices=False)
        
        # Store results
        self.spatial_modes = U  # Phi matrix
        self.singular_values = s
        self.temporal_coeffs = Vt.T
        
        # Calculate relative importance
        energy = s**2
        cumulative_energy = np.cumsum(energy) / np.sum(energy)
        
        print(f"First 10 singular values: {s[:10]}")
        print(f"Energy captured by first 5 modes: {cumulative_energy[4]*100:.2f}%")
        
        return U, s, Vt, cumulative_energy
    
    def plot_svd_analysis(self, U, s, cumulative_energy, n_modes=6):
        """Plot SVD analysis results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Singular values
        ax1.semilogy(s[:20], 'bo-')
        ax1.set_xlabel('Mode Number')
        ax1.set_ylabel('Singular Value')
        ax1.set_title('Singular Values (Log Scale)')
        ax1.grid(True)
        
        # Cumulative energy
        ax2.plot(cumulative_energy[:20], 'ro-')
        ax2.set_xlabel('Number of Modes')
        ax2.set_ylabel('Cumulative Energy')
        ax2.set_title('Energy Capture vs Number of Modes')
        ax2.grid(True)
        
        # Spatial modes
        colors = plt.cm.tab10(np.linspace(0, 1, n_modes))
        for i in range(min(n_modes, U.shape[1])):
            ax3.plot(self.x * 1000, U[:, i], color=colors[i], 
                    label=f'Mode {i+1}')
        ax3.set_xlabel('Position [mm]')
        ax3.set_ylabel('Mode Amplitude')
        ax3.set_title('Spatial Modes (Φ)')
        ax3.legend()
        ax3.grid(True)
        
        # Mode comparison
        modes_to_show = [0, 1, 2, 3]
        for i, mode in enumerate(modes_to_show):
            if mode < U.shape[1]:
                ax4.plot(self.x * 1000, U[:, mode], 
                        label=f'Mode {mode+1}', linewidth=2)
        ax4.set_xlabel('Position [mm]')
        ax4.set_ylabel('Mode Amplitude')
        ax4.set_title('First Few Spatial Modes')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def solve_reduced_order(self, source_func, n_modes, time_steps=100):
        """
        Solve using reduced order model with n_modes
        Following the approach from your friend's code
        """
        print(f"Solving reduced order model with {n_modes} modes...")
        
        if not hasattr(self, 'spatial_modes'):
            raise ValueError("Must perform SVD analysis first!")
        
        # Extract first n modes
        Phi = self.spatial_modes[:, :n_modes]
        
        # Compute derivatives of modes (like your friend's approach)
        dPhi_dx = np.zeros((self.nx, n_modes))
        d2Phi_dx2 = np.zeros((self.nx, n_modes))
        
        for i in range(n_modes):
            dPhi_dx[:, i] = np.gradient(Phi[:, i], self.x)
            d2Phi_dx2[:, i] = np.gradient(dPhi_dx[:, i], self.x)
        
        time_points = np.linspace(0, self.t_end, time_steps)
        snapshots_rom = np.zeros((self.nx, time_steps))
        b_hist = np.zeros((n_modes, time_steps))
        
        for i, t in enumerate(time_points):
            # Get source term
            source = source_func(t)
            
            # Solve for modal coefficients using least squares approach
            # d2Phi_dx2 * b = -source
            # Following your friend's approach: solve (d2Phi_dx2.T @ d2Phi_dx2) @ b = -d2Phi_dx2.T @ source
            A_modal = d2Phi_dx2.T @ d2Phi_dx2
            rhs_modal = -d2Phi_dx2.T @ source
            
            b = np.linalg.solve(A_modal, rhs_modal)
            b_hist[:, i] = b
            
            # Reconstruct solution: T = Phi * b
            T_rom = Phi @ b
            
            # Apply boundary condition T(L) = 0
            T_rom[-1] = 0
            
            snapshots_rom[:, i] = T_rom
        
        return snapshots_rom
    
    def compare_solutions(self, T_hr, T_rom, title="Solution Comparison"):
        """Compare high-resolution and ROM solutions"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot solutions at different times
        time_indices = [0, len(self.time_points)//3, 2*len(self.time_points)//3, -1]
        colors = ['blue', 'green', 'orange', 'red']
        
        for i, (idx, color) in enumerate(zip(time_indices, colors)):
            ax1.plot(self.x * 1000, T_hr[:, idx], '-', color=color, 
                    label=f't = {self.time_points[idx]:.1f}s (HR)', linewidth=2)
            ax1.plot(self.x * 1000, T_rom[:, idx], '--', color=color,
                    label=f't = {self.time_points[idx]:.1f}s (ROM)', linewidth=2)
        
        ax1.set_xlabel('Position [mm]')
        ax1.set_ylabel('Temperature [K]')
        ax1.set_title(f'{title} - Temperature Profiles')
        ax1.legend()
        ax1.grid(True)
        
        # Error analysis
        error = np.abs(T_hr - T_rom)
        relative_error = error / (np.abs(T_hr) + 1e-10)
        
        im1 = ax2.contourf(self.time_points, self.x * 1000, error, levels=20)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Position [mm]')
        ax2.set_title('Absolute Error |T_HR - T_ROM|')
        plt.colorbar(im1, ax=ax2)
        
        # Max error over time
        max_error = np.max(error, axis=0)
        ax3.plot(self.time_points, max_error, 'r-', linewidth=2)
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Max Absolute Error [K]')
        ax3.set_title('Maximum Error vs Time')
        ax3.grid(True)
        
        # RMS error over time
        rms_error = np.sqrt(np.mean(error**2, axis=0))
        ax4.plot(self.time_points, rms_error, 'b-', linewidth=2)
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('RMS Error [K]')
        ax4.set_title('RMS Error vs Time')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Maximum absolute error: {np.max(error):.2e} K")
        print(f"RMS error: {np.sqrt(np.mean(error**2)):.2e} K")

# Example usage and complete workflow
def run_complete_analysis():
    """Run the complete ROM analysis"""
    
    # Create solver
    solver = Heat1D_ROM(L=1.0, nx=200, t_end=1000)  # 1m length, 200 grid points
    
    print("="*60)
    print("1D HEAT EQUATION - REDUCED ORDER MODELING ANALYSIS")
    print("="*60)
    
    # Part a) High-resolution solution with tent distribution
    print("\n--- PART A: High-Resolution Solver ---")
    T_hr_tent = solver.solve_high_resolution(solver.tent_distribution, time_steps=50)
    
    # Part b) SVD Analysis
    print("\n--- PART B: SVD Analysis ---")
    U, s, Vt, cumulative_energy = solver.perform_svd(T_hr_tent)
    solver.plot_svd_analysis(U, s, cumulative_energy)
    
    # Part c) ROM with different numbers of modes
    print("\n--- PART C: Reduced Order Modeling ---")
    mode_counts = [2, 5, 10, 20]
    
    for n_modes in mode_counts:
        print(f"\nTesting ROM with {n_modes} modes:")
        T_rom = solver.solve_reduced_order(solver.tent_distribution, n_modes, time_steps=50)
        
        # Calculate error metrics
        error = np.abs(T_hr_tent - T_rom)
        max_error = np.max(error)
        rms_error = np.sqrt(np.mean(error**2))
        print(f"  Max error: {max_error:.2e} K")
        print(f"  RMS error: {rms_error:.2e} K")
        print(f"  DOFs: HR={solver.nx}, ROM={n_modes}, Reduction factor={solver.nx/n_modes:.1f}x")
    
    # Choose optimal number of modes (e.g., 10)
    n_modes_optimal = 10
    T_rom_tent = solver.solve_reduced_order(solver.tent_distribution, n_modes_optimal, time_steps=50)
    solver.compare_solutions(T_hr_tent, T_rom_tent, "Tent Distribution - ROM Validation")
    
    # Part d) Test with Hertzian distribution (different from training)
    print("\n--- PART D: Generalization Test ---")
    print("Testing ROM (trained on tent) with Hertzian distribution...")
    
    # High-resolution solution with Hertzian
    T_hr_hertz = solver.solve_high_resolution(solver.hertzian_distribution, time_steps=50)
    
    # ROM solution with Hertzian (using modes from tent distribution)
    T_rom_hertz = solver.solve_reduced_order(solver.hertzian_distribution, n_modes_optimal, time_steps=50)
    
    # Compare solutions
    solver.compare_solutions(T_hr_hertz, T_rom_hertz, "Hertzian Distribution - Generalization Test")
    
    print("\n--- ANALYSIS COMPLETE ---")
    print(f"Summary:")
    print(f"  High-resolution model: {solver.nx} unknowns")
    print(f"  ROM model: {n_modes_optimal} unknowns")
    print(f"  Computational reduction: {solver.nx/n_modes_optimal:.1f}x")
    print(f"  SVD modes capture {cumulative_energy[n_modes_optimal-1]*100:.1f}% of energy")

if __name__ == "__main__":
    run_complete_analysis()