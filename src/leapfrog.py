import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Leapfrog:
    def __init__(self, k=1.0, m=1.0, dt=0.01, T=10, x0=1.0, v0=0.0):
        self.k = k                              # Spring constant
        self.m = m                              # Mass
        self.dt = dt                            # Time step
        self.T = T                              # Total simulation time
        self.N = int(T / dt)                    # Number of time steps
        self.x = np.zeros(self.N + 1)           # Position values at full time-steps
        self.v = np.zeros(self.N + 1)           # Velocity values at full time-steps
        self.E = np.zeros(self.N + 1)           # Energy values at full time-steps

        # Initial conditions
        self.x[0] = x0
        self.v[0] = v0     

    def force(self, x):
        """Force function for the harmonic oscillator.
        Hooke's Law: F = -k * x
        """
        return -self.k * x
    
    def kinetic_energy(self, v):
        """Calculate the kinetic energy of the system for harmonic oscillator."""
        return 0.5 * self.m * v ** 2
    
    def potential_energy(self, x):
        """Calculate the potential energy of the system for harmonic oscillator."""
        return 0.5 * self.k * x ** 2
    
    def system_ode(self, t, y):
        """Define the system of ODEs for the harmonic oscillator."""
        self.x, self.v = y
        dv = self.force(self.x) / self.m
        return [self.v, dv]

    def driving_force(self, t, F0=1.0, omega=1.0):
        """Defines the external sinusoidal driving force F(t) = F0 * sin(omega * t)."""
        return F0 * np.sin(omega * t)

    def solve_leapfrog(self):
        """Leapfrog solver for the simple harmonic oscillator."""
        # Calculate initial half-step velocity: v(t+dt/2)
        initial_force = self.force(self.x[0])
        v_half = self.v[0] + 0.5 * self.dt * initial_force / self.m

        # Leapfrog integration loop
        for n in range(self.N):
            # Update position at half time-step
            self.x[n + 1] = self.x[n] + self.dt * v_half

            # Calculate new force and velocity at the next half-step
            new_force = self.force(self.x[n + 1])
            v_next_half = v_half + self.dt * new_force / self.m
            
            # Store the full-step velocity (approximation by averaging)
            self.v[n + 1] = (v_half + v_next_half) / 2
            
            # Update half-step velocity for next iteration
            v_half = v_next_half
            
            
        # Compute total energy
        self.E = 0.5 * self.kinetic_energy(self.v) + \
                 0.5 * self.potential_energy(self.x)
        
    def solve_leapfrog_with_forcing(self, F0=1.0, omega=1.0):
        """Leapfrog solver with an external sinusoidal driving force"""
        # Compute initial half-step velocity
        initial_force = self.force(self.x[0]) + self.driving_force(0, F0, omega)
        v_half = self.v[0] + 0.5 * self.dt * initial_force / self.m

        # Leapfrog integration loop
        for n in range(self.N):
            t_n = n * self.dt  # Current time

            # Update position
            self.x[n + 1] = self.x[n] + self.dt * v_half

            # Compute new force and velocity at the next half-step
            new_force = self.force(self.x[n + 1]) + self.driving_force(t_n, F0, omega)
            v_next_half = v_half + self.dt * new_force / self.m

            # Store the full-step velocity and update half-step velocity
            self.v[n + 1] = (v_half + v_next_half) / 2
            v_half = v_next_half

        # Compute total energy
        self.E = 0.5 * self.kinetic_energy(self.v) + \
                 0.5 * self.potential_energy(self.x)


    def solve_rk45(self):
        """Solve the system of ODEs using the Runge-Kutta 4th order method."""
        # Solve the above equations using runge-kutta 4th order method using solve_ivp
        t_eval = np.linspace(0, self.T, self.N + 1)
        y0 = [self.x[0], self.v[0]]

        # Solve using RK45
        solution = solve_ivp(self.system_ode, t_span=[0, self.T], y0=y0, method='RK45',
                             t_eval=t_eval, rtol=1e-8)
        
        # Extract results
        self.x = solution.y[0]
        self.v = solution.y[1]
        time = solution.t

        # Calculate energy components
        ke_rk45 = self.kinetic_energy(self.v)
        pe_rk45 = self.potential_energy(self.x)
        self.E = ke_rk45 + pe_rk45

        return time


    def plot_results_leapfrog(self, savefig=False, with_forcing=False, omega=1.0):
        """Plot position, velocity, and energy using Leapfrog method."""
        plt.figure(figsize=(10, 6), dpi=300)

        time_full = np.arange(self.N + 1) * self.dt

        # Position vs Time
        plt.subplot(2, 1, 1)
        plt.plot(time_full, self.x, label="Position (x)", color="blue")
        plt.plot(time_full, self.v, label="Velocity (v)", color="red", linestyle="dashed")
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Amplitude", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.title("Leapfrog Integration: Simple Harmonic Oscillator")
        plt.legend()
        plt.grid()

        # Energy Conservation
        plt.subplot(2, 1, 2)
        plt.plot(time_full, self.E, label="Total Energy", color="green")
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Energy", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(0, 1.5*np.max(self.E))
        # plt.title("Energy Conservation in Leapfrog Method")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        if savefig:
            if with_forcing:
                filepath = f"results/lf/lf_forcing_{omega}.png"
            else:
                filepath = "results/lf/lf_integration.png"
            plt.savefig(filepath, bbox_inches="tight")
            
        plt.show()

    def plot_position_many_k(self, k_values=[1.0], savefig=False):
        """Plot position vs time for different spring constants."""
        plt.figure(figsize=(10, 4), dpi=300)

        time_full = np.arange(self.N + 1) * self.dt

        for k in k_values:
            self.k = k
            self.solve_leapfrog()
            plt.plot(time_full, self.x, label=f"k = {k}")

        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Position (x)", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.title("Position vs Time for Different Spring Constants")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        if savefig:
            plt.savefig("results/lf/lf_position_many_k.png", bbox_inches="tight")
            
        plt.show()

    def plot_velocity_many_k(self, k_values=[1.0], savefig=False):
        """Plot velocity vs time for different spring constants."""
        plt.figure(figsize=(10, 4), dpi=300)

        time_full = np.arange(self.N + 1) * self.dt

        for k in k_values:
            self.k = k
            self.solve_leapfrog()
            plt.plot(time_full, self.v, label=f"k = {k}")

        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Velocity (v)", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.title("Velocity vs Time for Different Spring Constants")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        if savefig:
            plt.savefig("results/lf/lf_velocity_many_k.png", bbox_inches="tight")
            
        plt.show()

    def plot_energy_many_k(self, k_values=[1.0], savefig=False):
        """Plot energy conservation for different spring constants."""
        plt.figure(figsize=(10, 4), dpi=300)

        time_full = np.arange(self.N + 1) * self.dt

        for k in k_values:
            self.k = k
            self.solve_leapfrog()
            plt.plot(time_full, self.E, label=f"k = {k}")

        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Energy", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.title("Energy vs Time for Different Spring Constants")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        if savefig:
            plt.savefig("results/lf/lf_energy_many_k.png", bbox_inches="tight")
            
        plt.show()

    def plot_compare_methods(self, savefig=False):
        """Compare leapfrog and RK45 methods in terms of energy conservation."""
        plt.figure(figsize=(10, 6), dpi=300)

        time_full = np.arange(self.N + 1) * self.dt

        # Solve using Leapfrog
        self.solve_leapfrog()

        # Plot Leapfrog results
        plt.subplot(2, 1, 1)
        plt.plot(time_full, self.E, label="Leapfrog", color="blue")
        
        avg_energy_lf = np.mean(self.E)     # Energy trend for leapfrog
        plt.plot((0, self.T), (avg_energy_lf, avg_energy_lf), "k--", label="Energy Trend")

        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Energy", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc="lower left")
        plt.grid()
        plt.tight_layout()

        # Solve using RK45
        time_rk45 = self.solve_rk45()
        
        # Plot RK45 results
        plt.subplot(2, 1, 2)
        plt.plot(time_rk45, self.E, label="RK45", color="red")
        
        # Energy trend for RK45
        z = np.polyfit(time_rk45, self.E, 1)
        p = np.poly1d(z)
        plt.plot(time_rk45, p(time_rk45), "k--", label="Energy Trend")
        
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Energy", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.title("Energy Conservation: Leapfrog vs RK45")
        plt.legend(loc="lower left")
        plt.grid()
        plt.tight_layout()

        if savefig:
            plt.savefig("results/lf/compare_methods.png", bbox_inches="tight")
            
        plt.show()

    def plot_phase_space(self, F0=1.0, omega_values=[0.8, 1.0, 1.2], savefig=False):
        """Plots the phase space (velocity vs. position) for different driving frequencies."""
        plt.figure(figsize=(8, 6), dpi=300)
        colors = ["red", "green", "blue"]

        for omega in omega_values:
            self.solve_leapfrog_with_forcing(F0, omega)
            plt.plot(self.x, self.v, label=f"ω = {omega}", color=colors.pop())

        plt.xlabel("Position (x)", fontsize=16)
        plt.ylabel("Velocity (v)", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.title("Phase Space Plot (Resonance Behavior)")
        plt.legend()
        plt.grid()

        if savefig:
            plt.savefig("results/lf/lf_forcing_phase_space.png", bbox_inches="tight")

        plt.show()