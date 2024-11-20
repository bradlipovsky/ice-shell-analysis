import numpy as np

def simulate(params):
    # Extract parameters from params dictionary
    K = params['K']        # Bulk modulus (Pa)
    G = params['G']        # Shear modulus (Pa)
    rho = params['rho']    # Fluid density (kg/m^3)
    g = params['g']
    alpha = params['alpha']   # Biot's coefficient
    S = params['S']        # Storage coefficient (1/Pa)
    k = params['k']        # Permeability (m^2)
    a = params['a']        # Inner radius (m)
    b = params['b']        # Outer radius (m)
    N_r = params['N_r']    # Number of spatial grid points
    N_t = params['N_t']    # Number of time steps
    T = params['T']        # Total simulation time (s)
    
    # Allow p_w function to be passed in params, with default as constant pressure
    if 'p_w' in params:
        p_w = params['p_w']
    else:
        # Function for prescribed well pressure at R = a
        def p_w(t):
            # Example: constant pressure
            return 1.0
    
    # Effective radial stress boundary conditions
    sigma_RR_prime_a = params.get('sigma_RR_prime_a', 0.0)  # default to 0.0
    sigma_RR_prime_b = params.get('sigma_RR_prime_b', 0.0)  # default to 0.0
    
    # Derived parameters
    gamma = rho * g
    D = k / gamma  # Hydraulic diffusivity (m^2/s)
    A_param = K - (2 / 3) * G
    B_param = K + (4 / 3) * G
    
    # Spatial grid
    R = np.linspace(a, b, N_r + 1)
    Delta_R = R[1] - R[0]
    
    # Temporal grid
    t = np.linspace(0, T, N_t + 1)
    Delta_t = t[1] - t[0]
    
    # Initialize arrays to store solution fields at each timestep
    u_all = np.zeros((N_t + 1, N_r + 1))         # Displacement
    epsilon_all = np.zeros((N_t + 1, N_r + 1))   # Volumetric strain
    p_all = np.zeros((N_t + 1, N_r + 1))         # Pressure
    
    # Initial conditions
    u_prev = np.zeros(N_r + 1)
    epsilon_prev = np.zeros(N_r + 1)
    p_prev = np.zeros(N_r + 1)
    
    # Store initial conditions
    u_all[0, :] = u_prev
    epsilon_all[0, :] = epsilon_prev
    p_all[0, :] = p_prev
    
    # Time-stepping loop
    for n in range(N_t):
        # Time at current step
        t_n = t[n]
        t_np1 = t[n + 1]
        
        # Total number of unknowns
        N_unknowns = 2 * (N_r + 1)
        
        # Initialize coefficient matrix and RHS vector
        A_matrix = np.zeros((N_unknowns, N_unknowns))
        d_vector = np.zeros(N_unknowns)
        
        # Assemble equations for each node
        for i in range(N_r + 1):
            # Indices in the global system
            u_i = i                    # Index for u_i^{n+1}
            p_i = N_r + 1 + i          # Index for p_i^{n+1}
            
            R_i = R[i]
            
            # Mechanical Equation
            if i == 0:
                # Boundary condition at R = a (sigma_RR_prime_a)
                A_matrix[u_i, u_i] = (B_param / Delta_R) - (2 * A_param / R_i)
                A_matrix[u_i, u_i + 1] = -B_param / Delta_R
                d_vector[u_i] = sigma_RR_prime_a  # Given boundary value
            elif i == N_r:
                # Boundary condition at R = b (sigma_RR_prime_b)
                A_matrix[u_i, u_i - 1] = -B_param / Delta_R
                A_matrix[u_i, u_i] = (B_param / Delta_R) + (2 * A_param / R_i)
                d_vector[u_i] = -sigma_RR_prime_b  # Given boundary value
            else:
                # Mechanical equilibrium equation
                a_u = (1 / Delta_R ** 2) + (1 / (R_i * Delta_R))
                b_u = -2 / Delta_R ** 2 + (2 / R_i ** 2)
                c_u = (1 / Delta_R ** 2) - (1 / (R_i * Delta_R))
                
                A_matrix[u_i, u_i - 1] = B_param * c_u
                A_matrix[u_i, u_i] = B_param * b_u - 2 * A_param / R_i ** 2
                A_matrix[u_i, u_i + 1] = B_param * a_u
                
                # Coupling with pressure gradient
                A_matrix[u_i, p_i + 1 - (N_r + 1)] = - (alpha / (2 * Delta_R))
                A_matrix[u_i, p_i - 1 - (N_r + 1)] = (alpha / (2 * Delta_R))
                
                d_vector[u_i] = 0.0  # No external body forces
                
            # Diffusion Equation
            if i == 0:
                # Pressure boundary condition at R = a
                A_matrix[p_i, p_i] = 1.0
                d_vector[p_i] = p_w(t_np1)
            elif i == N_r:
                # No-flow boundary condition at R = b
                A_matrix[p_i, p_i] = 1.0
                A_matrix[p_i, p_i - 1] = -1.0
                d_vector[p_i] = 0.0
            else:
                # Diffusion equation
                D_over_DeltaR2 = D / Delta_R ** 2
                D_over_Ri_DeltaR = D / (R_i * Delta_R)
                
                a_p = - D_over_DeltaR2 + D_over_Ri_DeltaR / 2
                b_p = S / Delta_t + 2 * D_over_DeltaR2
                c_p = - D_over_DeltaR2 - D_over_Ri_DeltaR / 2
                
                A_matrix[p_i, p_i - 1] = a_p
                A_matrix[p_i, p_i] = b_p
                A_matrix[p_i, p_i + 1] = c_p
                
                # Coupling with displacement (epsilon_i^{n+1})
                A_matrix[p_i, u_i - 1] = - (alpha / Delta_t) * (1 / (2 * Delta_R))
                A_matrix[p_i, u_i] = - (alpha / Delta_t) * (2 / R_i)
                A_matrix[p_i, u_i + 1] = - (alpha / Delta_t) * (1 / (2 * Delta_R))
                
                # Right-hand side
                d_vector[p_i] = (S / Delta_t) * p_prev[i] - (alpha / Delta_t) * epsilon_prev[i]
                
        # Solve the linear system
        solution = np.linalg.solve(A_matrix, d_vector)
        
        # Extract u and p from the solution vector
        u = solution[:N_r + 1]
        p = solution[N_r + 1:]
        
        # Compute epsilon at time n+1
        epsilon = np.zeros(N_r + 1)
        epsilon[1:-1] = (u[2:] - u[:-2]) / (2 * Delta_R) + (2 * u[1:-1]) / R[1:-1]
        epsilon[0] = (u[1] - u[0]) / Delta_R + (2 * u[0]) / R[0]
        epsilon[-1] = (u[-1] - u[-2]) / Delta_R + (2 * u[-1]) / R[-1]
        
        # Update variables for next time step
        u_prev = u.copy()
        epsilon_prev = epsilon.copy()
        p_prev = p.copy()
        
        # Save the solution fields at this timestep
        u_all[n + 1, :] = u
        epsilon_all[n + 1, :] = epsilon
        p_all[n + 1, :] = p
        
    # Return the solution arrays and the grids
    return u_all, epsilon_all, p_all, t, R