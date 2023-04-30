from .meta import *
from .common import *

from scipy.integrate import solve_ivp

class HomogeneousDiffSEIR_2groups_2mode:
    stochastic = False
    parameters = [
        'pinf',
    ] + [
        f'q{i}{j}'
        for i in range(1,3)
        for j in range(1,3)
    ]
    
    def poop():

        def dinf_compartments(self, t,y):
            ia,ea,ra,ib,eb,rb = y
            sa = 1-ia-ea-ra
            sb = 1-ib-eb-rb

            return (
                # diA/dt
                ea*B1 - ia*B2,
                # deA/dt
                ia*sa*q[1,1,1] + ib*sa*q[1,2,1] + ia*sa*q[1,1,2] + ib*sa*q[1,2,2] - ea*B1,
                # drA/dt
                ia*B2,

                # diB/dt
                eb*B1 - ib*B2,
                # deB/dt
                ib*sb*q[2,2,1] + ia*sb*q[1,2,1] + ib*sb*q[2,2,2] + ia*sb*q[1,2,2] - eb*B1,
                #drB/dt
                ib*B2,
            )

        #y0 = (0.01, 0)
        y0 = (0.5, 0, 0, 0, 0, 0)
        t_span = [0,300]

        sol = solve_ivp(self.dinf_compartments, t_span, y0, 
                  method='RK45', t_eval=np.linspace(0,300,200), dense_output=False, 
                  events=None, vectorized=False, args=None)

class HomogeneousDiffSI(Model):
    stochastic = False
    parameters = [
        'beta_si',
        'T',
        'frac_inf_0',
    ]
    
    def simulate(self, params):

        def dinf(t,y):
            ia = y
            sa = 1-ia
            return (
                # di/dt
                ia*sa* params.beta_si
            )

        #y0 = (0.01, 0)
        y0 = (params.frac_inf_0,)
        t_span = [0, params.T]

        sol = solve_ivp(dinf, t_span, y0, 
                  method='RK45', t_eval=np.linspace(0,params.T,200), dense_output=False, 
                  events=None, vectorized=False, args=None)
        return Results(
            frac_i_daily= sol.y[0],
            frac_s_daily= 1-sol.y[0],
        )

class HomogeneousDiffSIR(Model):
    stochastic = False
    parameters = [
        'beta_si',
        'beta_ir',
        'T',
        'frac_i_0',
        'frac_r_0'
    ]
    
    def simulate(self, params):

        def dinf(t,y):
            ia,ra = y
            sa = 1-ia-ra
            return (
                # di/dt
                ia*sa* params.beta_si - params.beta_ir * ia,
                # dr/dt
                params.beta_ir * ia,
            )

        #y0 = (0.01, 0)
        y0 = (params.frac_i_0, params.frac_r_0)
        t_span = [0, params.T]

        sol = solve_ivp(dinf, t_span, y0, 
                  method='RK45', t_eval=range(0,params.T+1), dense_output=False, 
                  events=None, vectorized=False, args=None)
        return Results(
            frac_i_daily= sol.y[0],
            frac_r_daily= sol.y[1],
            frac_s_daily= 1-sol.y[0]-sol.y[1],
        )