import numpy as np
import ot


class KSC:
    def __init__(self, Kcc, Kct, Ktt, l, v=None, w=None):
        # H (n, n) symmetric
        # C (n, m)
        # l = lambda
        # v (m,)
        # w (n,)

        self.Kcc = Kcc
        self.Kct = Kct
        self.Ktt = Ktt
        self.l = l

        self.n, self.m = np.shape(Kct)

        if v is None:
            self.v = np.ones(self.m) / self.m
        else:
            self.v = v

        if w is None:
            self.w = np.ones(self.n) / self.n
        else:
            self.w = w

        self.Linf = np.max(np.abs(Kcc))    # L_infinity norm of Kcc
        self.vmin = np.min(self.v)
        self.quadconst = 0.5*np.trace(self.v*self.Ktt)

    # quadratic function g = 0.5 * <pi / v, Kcc pi> - <Kct, pi> + 0.5 * tr(v * Ktt)
    def g(self, pi):
        return 0.5*np.sum(np.multiply(pi/self.v, self.Kcc@pi)) - np.sum(np.multiply(self.Kct, pi)) + self.quadconst
    
    # gradient of g = Kcc pi / v - Kct
    def g_grad(self, pi):
        return (self.Kcc@pi/self.v - self.Kct)

    # entropy term
    def h(self, pi):
        return np.sum(np.multiply(pi, np.log(pi / np.e)))
    
    # fixed-point update
    def fp_update(self, pi):
        pi_next = ot.sinkhorn(self.w, self.v, self.g_grad(pi), reg=self.l)
        return pi_next
    
    # Gradient descent with KL divergence
    def gdkl_update(self, pi, tau):
        pi_next = ot.sinkhorn(self.w, self.v, self.g_grad(pi)+(self.l-(1.0/tau))*np.log(pi), reg=(1.0/tau))
        return pi_next
    
    # solve the optimization problem
    def solve(self, method, tau_scale=1.0, max_iter=10000, pi_tol=1e-5, obj_tol=1e-16, pi_init=None, verbose=False):

        # initialization
        if pi_init is None:
            pi = self.w[:,np.newaxis] @ self.v[np.newaxis,:]
            
        else:
            pi = pi_init

        # compute the objective
        obj = self.g(pi) + self.l*self.h(pi)

        if verbose==True:
            print(f'Initialization: objective = {obj}')

        # fixed point
        if method == 'fp':
            print('Implement fixed point algorithm')
            
            # iterations
            i = 0
            while i < max_iter:

                pi_prev = pi
                obj_prev = obj

                ###################################
                # implement the optimization update
                pi = self.fp_update(pi_prev)
                ###################################

                # compute the change in pi (L1)    
                pi_change = np.sum(np.abs(pi - pi_prev))  # L1 distance

                # compute the relative change in the objective value
                obj = self.g(pi) + self.l*self.h(pi)
                obj_change = obj - obj_prev
                obj_rel_change = obj_change / np.abs(obj_prev)

                ###################################
                # method-specific rules
                if obj_rel_change > 0.0: 
                    print(f"Terminated after {i} iterations (objective increased)")
                    pi = pi_prev    # no update and return the previous pi
                    break
                ###################################

                # print the results
                if verbose==True:
                    print(f'Iteration {i + 1}: objective = {obj}, objective relative change = {obj_rel_change}, pi change (L1) = {pi_change}')

                ###################################
                # method-specific stopping criteria
                # stop if the objective does not change = obj(pi) is close to obj(pi_prev)
                if np.abs(obj_rel_change) < obj_tol:
                    print(f"Terminated after {i + 1} iterations (objective converged)")
                    break

                # stop if pi does not change = pi is close to pi_prev in L1
                if pi_change < pi_tol:
                    print(f"Terminated after {i + 1} iterations (coupling converged)")
                    break
                ###################################

                # to the next iteration
                i += 1

        # Gradient descent with KL divergence            
        elif method == 'gdkl':
            print('Implement gradient descent with KL divergence')
            tau = tau_scale / (self.Linf/self.vmin + self.l)

            # iterations
            i = 0
            while i < max_iter:

                pi_prev = pi
                obj_prev = obj

                ###################################
                # implement the optimization update
                pi = self.gdkl_update(pi_prev, tau)
                ###################################

                # compute the change in pi (L1)    
                pi_change = np.sum(np.abs(pi - pi_prev))  # L1 distance

                # compute the relative change in the objective value
                obj = self.g(pi) + self.l*self.h(pi)
                obj_change = obj - obj_prev
                obj_rel_change = obj_change / np.abs(obj_prev)
                
                ###################################
                # method-specific rules
                ###################################
                
                # print the results
                if verbose==True:
                    print(f'Iteration {i + 1}: objective = {obj}, objective relative change = {obj_rel_change}, pi change (L1) = {pi_change}')

                ###################################
                # method-specific stopping criteria
                # stop if the objective does not change = obj(pi) is close to obj(pi_prev)
                if np.abs(obj_rel_change) < obj_tol:
                    print(f"Terminated after {i + 1} iterations (objective converged)")
                    break

                # stop if pi does not change = pi is close to pi_prev in L1
                if pi_change < pi_tol:
                    print(f"Terminated after {i + 1} iterations (coupling converged)")
                    break
                ###################################

                # to the next iteration
                i += 1
        
        else:
            raise ValueError('Invalid method')

        if i == max_iter:
            print("Reached the max iteration")

        # save and return the solution
        self.pi = pi
        return pi