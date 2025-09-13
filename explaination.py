import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def heat_equation_explanation():
    """
    Step-by-step explanation of solving the 1D heat equation
    """
    print("="*60)
    print("1D HEAT EQUATION SOLUTION - STEP BY STEP")
    print("="*60)
    
    print("PROBLEM STATEMENT:")
    print("Solve: d²T/dx² + Ex/k = 0")
    print("Boundary conditions:")
    print("  Left (x=0):  dT/dx = 0  (insulated)")
    print("  Right (x=L): T = 0      (fixed temperature)")
    print()
    
    # Setup
    L = 1.0  # Length [m]
    nx = 11  # Small number of points for clarity
    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]
    
    print(f"Domain: 0 to {L}m")
    print(f"Grid points: {nx}")
    print(f"Grid spacing: dx = {dx:.2f}m")
    print(f"Grid: {[f'{xi:.1f}' for xi in x]}")
    print()
    
    # STEP 1: FINITE DIFFERENCE APPROXIMATION
    print("="*40)
    print("STEP 1: FINITE DIFFERENCE APPROXIMATION")
    print("="*40)
    
    print("We approximate the second derivative using finite differences:")
    print("d²T/dx² ≈ (T[i-1] - 2*T[i] + T[i+1]) / dx²")
    print()
    print("So our equation becomes:")
    print("(T[i-1] - 2*T[i] + T[i+1]) / dx² + Ex[i]/k = 0")
    print()
    print("Rearranging:")
    print("T[i-1] - 2*T[i] + T[i+1] = -Ex[i]*dx²/k")
    print()
    
    # STEP 2: BOUNDARY CONDITIONS
    print("="*40)
    print("STEP 2: HANDLING BOUNDARY CONDITIONS")
    print("="*40)
    
    print("LEFT BOUNDARY (x=0): dT/dx = 0")
    print("We use the 'ghost point' method:")
    print("dT/dx ≈ (T[1] - T[-1])/(2*dx) = 0")
    print("This means: T[-1] = T[1] (ghost point)")
    print()
    print("For point i=0:")
    print("T[-1] - 2*T[0] + T[1] = -Ex[0]*dx²/k")
    print("T[1] - 2*T[0] + T[1] = -Ex[0]*dx²/k  (substitute T[-1] = T[1])")
    print("2*T[1] - 2*T[0] = -Ex[0]*dx²/k")
    print()
    
    print("RIGHT BOUNDARY (x=L): T = 0")
    print("We simply fix T[n-1] = 0")
    print("So we don't need to solve for the last point!")
    print()
    
    # STEP 3: MATRIX FORMULATION
    print("="*40)
    print("STEP 3: MATRIX FORMULATION")
    print("="*40)
    
    print("We solve for temperature at internal points (excluding right boundary)")
    print(f"Number of unknowns: {nx-1} (points 0 to {nx-2})")
    print()
    
    # Build matrix manually to show structure
    n = nx - 1  # Number of unknowns
    A = np.zeros((n, n))
    
    print("Building the coefficient matrix A:")
    print("Each row represents the finite difference equation for one grid point")
    print()
    
    # Fill the matrix
    for i in range(n):
        if i == 0:
            # First point (left boundary)
            A[i, i] = -2    # Coefficient of T[0]
            A[i, i+1] = 2   # Coefficient of T[1] (note: 2, not 1!)
            print(f"Row {i} (x={x[i]:.1f}): -2*T[{i}] + 2*T[{i+1}] = -Ex[{i}]*dx²")
        elif i == n-1:
            # Last unknown point (next to right boundary)
            A[i, i-1] = 1   # Coefficient of T[i-1]
            A[i, i] = -2    # Coefficient of T[i]
            # T[i+1] = 0 (boundary condition), so no column for it
            print(f"Row {i} (x={x[i]:.1f}): T[{i-1}] - 2*T[{i}] + 0 = -Ex[{i}]*dx²")
        else:
            # Interior points
            A[i, i-1] = 1   # Coefficient of T[i-1]
            A[i, i] = -2    # Coefficient of T[i]
            A[i, i+1] = 1   # Coefficient of T[i+1]
            print(f"Row {i} (x={x[i]:.1f}): T[{i-1}] - 2*T[{i}] + T[{i+1}] = -Ex[{i}]*dx²")
    
    # Scale by dx²
    A = A / (dx**2)
    
    print(f"\nCoefficient matrix A (scaled by 1/dx² = {1/dx**2:.2f}):")
    print(A)
    print()
    
    # STEP 4: SOLVE EXAMPLE
    print("="*40)
    print("STEP 4: SOLVE AN EXAMPLE")
    print("="*40)
    
    # Create a simple heat source (tent-shaped)
    Ex_k = np.zeros(nx)
    center = nx // 2
    width = 3
    max_source = 1000
    
    for i in range(nx):
        if abs(i - center) <= width:
            Ex_k[i] = max_source * (1 - abs(i - center) / width)
    
    print("Heat source Ex/k:")
    for i in range(nx):
        print(f"x[{i}] = {x[i]:.1f}m: Ex/k = {Ex_k[i]:.1f}")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, Ex_k, 'ro-')
    plt.xlabel('Position [m]')
    plt.ylabel('Heat Source Ex/k [K/m²]')
    plt.title('Heat Source Distribution')
    plt.grid(True)
    
    # Right-hand side vector
    rhs = -Ex_k[:-1]  # Remove last point (boundary condition)
    print(f"\nRight-hand side vector (RHS = -Ex/k):")
    for i in range(len(rhs)):
        print(f"RHS[{i}] = {rhs[i]:.1f}")
    
    # Solve the system
    print(f"\nSolving the system: A * T = RHS")
    A_sparse = csr_matrix(A)
    T_internal = spsolve(A_sparse, rhs)
    
    # Reconstruct full temperature field
    T_full = np.zeros(nx)
    T_full[:-1] = T_internal
    T_full[-1] = 0  # Right boundary condition
    
    print(f"\nSolution:")
    for i in range(nx):
        print(f"T[{i}] at x={x[i]:.1f}m: {T_full[i]:.1f} K")
    
    plt.subplot(1, 2, 2)
    plt.plot(x, T_full, 'bo-')
    plt.xlabel('Position [m]')
    plt.ylabel('Temperature [K]')
    plt.title('Temperature Solution')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # STEP 5: VERIFY THE SOLUTION
    print("="*40)
    print("STEP 5: VERIFY THE SOLUTION")
    print("="*40)
    
    print("Let's check if our solution satisfies the original equation...")
    print("We need: d²T/dx² + Ex/k = 0")
    print()
    
    # Compute second derivative numerically
    d2T_dx2 = np.zeros(nx)
    for i in range(1, nx-1):
        d2T_dx2[i] = (T_full[i-1] - 2*T_full[i] + T_full[i+1]) / (dx**2)
    
    # Handle boundaries
    d2T_dx2[0] = (T_full[1] - 2*T_full[0] + T_full[1]) / (dx**2)  # Ghost point
    d2T_dx2[-1] = (T_full[-2] - 2*T_full[-1] + 0) / (dx**2)       # Right BC
    
    residual = d2T_dx2 + Ex_k
    
    print("Verification (should be ≈ 0):")
    for i in range(nx):
        print(f"x[{i}] = {x[i]:.1f}m: d²T/dx² + Ex/k = {d2T_dx2[i]:.1f} + {Ex_k[i]:.1f} = {residual[i]:.2f}")
    
    print(f"\nMaximum residual: {np.max(np.abs(residual)):.2e}")
    print("(Should be very close to zero)")
    
    # STEP 6: BOUNDARY CONDITION CHECK
    print("\n" + "="*40)
    print("STEP 6: BOUNDARY CONDITION CHECK")
    print("="*40)
    
    # Left boundary: dT/dx = 0
    dT_dx_left = (T_full[1] - T_full[0]) / dx  # Forward difference
    print(f"Left BC check:")
    print(f"dT/dx at x=0: {dT_dx_left:.2e} (should be ≈ 0)")
    
    # Right boundary: T = 0
    print(f"Right BC check:")
    print(f"T at x=L: {T_full[-1]:.2e} (should be = 0)")
    
    # Visual representation of the method
    print("\n" + "="*40)
    print("SUMMARY: WHAT WE DID")
    print("="*40)
    
    print("1. Discretized the domain into grid points")
    print("2. Approximated d²T/dx² using finite differences")
    print("3. Applied boundary conditions:")
    print("   - Left: dT/dx = 0 → used ghost point method")
    print("   - Right: T = 0 → fixed value")
    print("4. Set up linear system A*T = -Ex/k")
    print("5. Solved for temperature at each grid point")
    print()
    print("The key insight: The differential equation becomes")
    print("a system of linear algebraic equations!")
    
    return x, T_full, Ex_k, A

def compare_with_friend_code():
    """
    Show how this relates to your friend's implementation
    """
    print("\n" + "="*60)
    print("COMPARISON WITH YOUR FRIEND'S CODE")
    print("="*60)
    
    print("Your friend's matrix setup:")
    print("```python")
    print("A = np.zeros((ndofs-1,ndofs-1))")
    print("np.fill_diagonal(A, -2)")
    print("np.fill_diagonal(A[:, 1:], 1)")
    print("np.fill_diagonal(A[1:,:], 1)")
    print("A[0,0] = -2")
    print("A[0, 1] = 2  # This is the key for left BC!")
    print("A = A/(dx**2)")
    print("```")
    print()
    print("This creates the same matrix we built manually!")
    print("The key insight is A[0,1] = 2 instead of 1")
    print("This implements the ghost point method for dT/dx = 0")

if __name__ == "__main__":
    x, T, Ex_k, A = heat_equation_explanation()
    compare_with_friend_code()