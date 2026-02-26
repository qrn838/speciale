import numpy as np
import pandas as pd
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d
from scipy.stats import beta

from funcs import *

class sicknessleavemodel(EconModelClass):

    def settings(self):
        """ fundamental settings """

        self.namespaces = ['par', 'sol', 'sim', 'data'] 

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par
        data = self.data

        par.after_reform = 0 # 0 = before reform, 1 = after reform, 2 = simulation 1, 3 = simulation 2

        par.T = 200 # time periods on sickness leave
        par.T_emp = 200 # time periods reemployed
        
        # preferences
        par.beta = 0.95                  # Discount factor

        # health process
        par.rho = 0.915                   # Persistence parameter
        par.sigma = 1.969                  # Standard deviation of the shock
        par.mu = 0.0                     # Mean of the shock
        par.m = 20                        # Number of grid points for the shock  

        par.alpha = 2.574                  # Initital health distribution parameter
        par.eta = 2.213                    # Initital health distribution parameter

        par.kappa_1 = 0.547                # Cost of working
        par.kappa_2 = 0.968                # Cost of working
        par.kappa_3 = 0.644 # Cost of working
        par.gamma = 1.025                  # Cost of working curvature
        par.eps = 1e-6                   # Small number

        # income
        par.b_sick = 0.546 #4900.0 #0.8                # Benefit level when sick
        par.b_reduced_1 = 0.388
        par.b_reduced_0 = 0.0

        # Probility of benefit reduction
        # For lower benefit: quadratic with a negative quadratic coefficient.
        par.delta0_1 = -0.838
        par.delta1_1 = 0.898
        par.delta2_1 = -0.322  # negative to force a peak at intermediate h

        # For zero benefit: a linear (or mildly quadratic) increasing function.
        par.delta0_2 = -0.985
        par.delta1_2 = 6.519
        par.delta2_2 = 0.163  # set to 0 for a linear effect

        par.w = 1.0 #9000.0                     # True wage
        par.xi = 0.00                     # Wage bias

        # grids  
        par.n = 200                      # number of grid points for health

        # simulation
        par.simN = 100000 # number of individuals

        # Parameters for VFI
        par.max_iter = 10000   # maximum number of iterations
        par.tol = 1.0e-8 #convergence tol. level 

        # Data for estimation
        par.full_sample_estimation = True

        data.data = pd.read_csv('hazard.csv', header=None)
        data.hazard_before = data.data[2:96]
        data.hazard_after = data.data[98:]
        # make the indixes the same
        data.hazard_before.index = range(7, len(data.hazard_before)+7)
        data.hazard_after.index = range(7, len(data.hazard_after)+7)

        data.weight = pd.read_csv('weight.csv', header=None)
        data.weight_before = np.linalg.inv(np.diagflat(np.array(data.weight[2:96])))
        data.weight_after = np.linalg.inv(np.diagflat(np.array(data.weight[98:])))

        data.rows_before, data.cols_before = data.weight_before.shape
        data.rows_after, data.cols_after = data.weight_after.shape
        
        data.weight_mat = np.zeros((data.rows_before + data.rows_after, data.cols_before + data.cols_after))
        data.weight_mat[:data.rows_before, :data.cols_before] =  data.weight_before
        data.weight_mat[data.rows_after:, data.cols_after:] = data.weight_after  

        data.sample_size = pd.read_csv('samplesize.csv', header=None)
        data.sample_size_before = data.sample_size[:101]
        data.sample_size_after = data.sample_size[101:]

        # Initial guesses for estimation
        par.lb_rho = 0.8
        par.ub_rho = 0.99
        par.lb_sigma = 0.5
        par.ub_sigma = 1.0
        par.lb_beta = 0.8
        par.ub_beta = 0.99
        par.lb_kappa = 0.5
        par.ub_kappa = 2.0
        par.lb_gamma = 0.1
        par.ub_gamma = 1.0
        par.lb_alpha = 1.0
        par.ub_alpha = 2.0
        par.lb_eta = 1.0
        par.ub_eta = 2.0
        par.lb_delta0_1 = -3.0
        par.ub_delta0_1 = 3.0
        par.lb_delta1_1 = 0.0
        par.ub_delta1_1 = 3.0
        par.lb_delta2_1 = -0.5
        par.ub_delta2_1 = 0.0
        par.lb_delta0_2 = -3.0
        par.ub_delta0_2 = 3.0
        par.lb_delta1_2 = 0.0
        par.ub_delta1_2 = 3.0
        par.lb_delta2_2 = 0.0
        par.ub_delta2_2 = 0.5

    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        if par.after_reform == 1:
            # par.t_c = np.array([30])
            par.t_c = np.array([28,29,30,31,32])                     # Time period when the benefit is reduced
            par.b_reduced = par.b_reduced_1 #3500.0 #0.6              # Reduced benefit level
        elif par.after_reform == 0:
            # par.t_c = np.array([62])
            par.t_c = np.array([60,61,62,63,64])                     # Time period when the benefit is reduced
            par.b_reduced = par.b_reduced_0              # Reduced benefit level
        elif par.after_reform == 2:
            # par.t_c = np.array([30])
            par.t_c = np.array([28,29,30,31,32])
            par.b_reduced = 0.0
        else:
            # par.t_c = np.array([62])
            par.t_c = np.array([60,61,62,63,64])
            par.b_reduced = 0.388 #3500.0 #0.6
        
        par.t_c_size = len(par.t_c)

        par.kappa = np.array([par.kappa_1, par.kappa_2, par.kappa_3])  # Cost of working
        par.kappa_size = len(par.kappa)

        par.simT = par.T

        par.bias = par.xi * par.w                   # Wage bias

        # Health grid
        par.grid = discretize_health_logistic_AR1(par)[0]
        par.transition = discretize_health_logistic_AR1(par)[1]

        # Benefit grid
        par.r_grid = np.array([0, 1, 2])
        par.r_size = len(par.r_grid)

        # Solution arrays
        shape = (par.T,par.n,par.r_size,par.t_c_size,par.kappa_size)
        sol.choice = np.nan + np.zeros(shape)
        sol.V_sick = np.nan + np.zeros(shape)
        sol.V_work = np.nan + np.zeros(shape)

        # Simulation arrays
        shape_sim = (par.simN,par.simT)
        sim.h = np.nan + np.zeros(shape_sim)
        sim.b = np.nan + np.zeros(shape_sim)
        sim.choice = np.zeros(shape_sim,dtype=np.int_)
        sim.choice_het = np.zeros((par.simN, par.simT, par.r_size),dtype=np.int_)
        sim.r = np.zeros(shape_sim,dtype=np.int_)
        sim.hazard = np.zeros(par.simT-1)
        sim.hazard_het = np.zeros((par.simT-1, par.r_size))
        sim.exit_health = np.nan + np.zeros(par.simN)

        sim.t_c = np.zeros(shape_sim,dtype=np.int_)

        # Draws used to simulate benefit reduction
        np.random.seed(9210)
        sim.draws_uniform = np.random.uniform(size=shape_sim)

        # Draws used to simulate health transitions
        sim.draws_normal = np.random.normal(loc=0.0, scale=par.sigma, size=shape_sim)

        # Initial states
        pdf_vals = beta.pdf(par.grid, par.alpha, par.eta)

        # 2) Normalize so it sums to 1
        pdf_vals /= pdf_vals.sum()

        # 3) Sample initial health states
        sim.h_init = np.random.choice(par.grid, size=par.simN, p=pdf_vals)



    ############
    # Solution #
    def solve(self):

        # unpack
        par = self.par
        sol = self.sol
        
        # solve for value of reemployment
        Value_work_ss = solve_steady_state_work(par)
        sol.V_work = value_of_reemployment(par, Value_work_ss)

        # Solve for steady-state sickness leave values & policy function
        V_S_full_ss, policy_S_full_ss = solve_steady_state_sickness_optimal_HtM(par, sol.V_work, 0)  # Full benefits
        V_S_reduced_ss, policy_S_reduced_ss = solve_steady_state_sickness_optimal_HtM(par, sol.V_work, 1)  # Reduced benefits
        V_S_out_ss, policy_S_out_ss = solve_steady_state_sickness_optimal_HtM(par, sol.V_work, 2)  # No benefits

        V_sick_ss = np.zeros((par.n,par.r_size,par.t_c_size,par.kappa_size), dtype=np.float_)
        V_sick_ss[:,0,:,:] = V_S_full_ss
        V_sick_ss[:,1,:,:] = V_S_reduced_ss
        V_sick_ss[:,2,:,:] = V_S_out_ss

        Policy_ss = np.zeros((par.n,par.r_size,par.t_c_size,par.kappa_size), dtype=np.int_)
        Policy_ss[:,0,:,:] = policy_S_full_ss
        Policy_ss[:,1,:,:] = policy_S_reduced_ss
        Policy_ss[:,2,:,:] = policy_S_out_ss

        # loop backwards (over all periods)
        for t in reversed(range(par.T)):

            # loop over state variables: health and benefit reduction
            for i_h,h in enumerate(par.grid):
                for i_r,r in enumerate(par.r_grid):
                    for i_t, tc in enumerate(par.t_c):
                        for i_k, k in enumerate(par.kappa):
                            idx = (t,i_h,i_r,i_t,i_k)
                            if t==par.T-1: # last period
                                
                                sol.V_sick[idx] = V_sick_ss[i_h,i_r,i_t,i_k]
                                sol.choice[idx] = Policy_ss[i_h,i_r,i_t,i_k]

                            else:
                                
                                # Utility from sick leave benefits
                                u_s = utility_t(0, h, t, tc, r, k, par)

                                # Expected continuation value IF staying on sick leave
                                EV_sick = np.dot(
                                                    par.transition[i_h, :],
                                                        np.maximum(
                                                        sol.V_sick[t+1,:,i_r,i_t,i_k],  # staying on sick leave next period
                                                        sol.V_work[:,i_k]           # switching to work next period
                                                    )
                                                )
                                sol.V_sick[idx] = u_s + par.beta * EV_sick

                                if sol.V_sick[idx] >= sol.V_work[i_h,i_k]:
                                    sol.choice[idx] = 0  # Stay on sick leave
                                else:
                                    sol.choice[idx] = 1  # Return to work
        
    ##############
    # Simulation #
    def simulate(self):

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        idx_h = np.zeros((par.simN,par.simT),dtype=np.int_)
        idx_t_c = np.zeros(par.simN,dtype=np.int_)
        t_c = np.zeros(par.simN,dtype=np.int_)
        k = np.zeros(par.simN,dtype=np.int_)

        # loop over individuals and time
        for i in range(par.simN):

            # initialize states
            sim.h[i,0] = sim.h_init[i]
            idx_h[i,0] = np.where(par.grid == sim.h[i,0])[0][0]
            sim.r[i,0] = 0
            sim.b[i,0] = par.b_sick
            j = np.random.randint(par.t_c_size)     # pick an integer index 0…K-1
            t_c[i] = par.t_c[j]    # the actual draw
            idx_t_c[i] = j             # the index in par.t_c
            k[i] = np.random.randint(par.kappa_size)     # pick an integer index 0…K-1

            for t in range(par.simT):

                # Save state specific values choice
                idx_sol = (t,idx_h[i,t],sim.r[i,t],idx_t_c[i],k[i])
                sim.choice[i,t] = sol.choice[idx_sol]
                if t>0 :
                    if sim.choice[i,t-1] == 1:
                        sim.choice[i,t] = 1
                    if sim.choice[i,t-1] == 0 and sim.choice[i,t] == 1:
                        sim.exit_health[i] = sim.h[i,t]
                if t==0:
                    if sim.choice[i,t] == 1:
                        sim.exit_health[i] = sim.h[i,t]


                # Store next-period states
                if t<par.simT-1:

                    idx_h[i,t+1] = draw_next_state(idx_h[i,t],par)
                    sim.h[i,t+1] = par.grid[idx_h[i,t+1]]

                    if t<t_c[i]-1:
                        sim.r[i,t+1] = sim.r[i,t]
                        sim.b[i,t+1] = sim.b[i,t]

                    elif t==t_c[i]-1:
                        if sim.draws_uniform[i,t] < delta_t(sim.h[i,t+1],par)[1]:
                            sim.r[i,t+1] = 1
                            sim.b[i,t+1] = par.b_reduced
                            sim.choice_het[i,0:t,sim.r[i,t+1]] = sim.choice[i,0:t]
                        elif sim.draws_uniform[i,t] < delta_t(sim.h[i,t+1],par)[2]:
                            sim.r[i,t+1] = 2
                            sim.b[i,t+1] = 0
                            sim.choice_het[i,0:t,sim.r[i,t+1]] = sim.choice[i,0:t]
                        else:
                            sim.r[i,t+1] = 0
                            sim.b[i,t+1] = par.b_sick
                            sim.choice_het[i,0:t,sim.r[i,t+1]] = sim.choice[i,0:t]
                    
                    else:
                        sim.r[i,t+1] = sim.r[i,t]
                        sim.b[i,t+1] = sim.b[i,t]
                        sim.choice_het[i,t,sim.r[i,t+1]] = sim.choice[i,t]
        
        sim.hazard[:] = compute_hazard(sim.choice[:,:])
        sim.hazard_het[:,0] = compute_hazard(sim.choice_het[:,:,0])
        sim.hazard_het[:,1] = compute_hazard(sim.choice_het[:,:,1])
        sim.hazard_het[:,2] = compute_hazard(sim.choice_het[:,:,2])


                
                    


