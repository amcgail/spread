from .meta import *
from .common import *

class EventNetworkDiscreteTimeSIR(Model):
    
    S = 0
    E = 1
    I = 2
    R = 3
    
    parameters = [
        'events', # node1, node2, time
        'bern_si',
        'bern_ir',
        'T',
        'frac_i_0',
        'frac_r_0'
    ]
    
    def __init__(self):
        self.m = defaultdict(list)
        self.t = 0
        
        
    def _msave(self):
        self.m['S'].append( np.sum(self.X==self.S) )
        self.m['I'].append( np.sum(self.X==self.I) )
        self.m['R'].append( np.sum(self.X==self.R) )
        
        # do I need the more detailed state?
        # self.m['X'].append( self.X.copy() )
        
    def simulate(self, params):        
        # rearranging for efficient access
        self.tsex = defaultdict(list)
        self.nodes = set()
        for t,a,b in params.events:
            self.nodes.update({a,b})
            
        self.nodes = sorted(self.nodes)
        self.n2i = {n:i for i,n in enumerate(self.nodes)}
        self.i2n = {i:n for i,n in enumerate(self.nodes)}
        
        # converting these names to mere integers
        for t,a,b in params.events:
            self.tsex[t].append((self.n2i[a],self.n2i[b]))
            
        self.N = len(self.nodes)
        
        self.X = np.array([self.S for _ in range(self.N)])
        
        for i in sample(range(self.N), int(params.frac_i_0*self.N)):
            self.X[i] = self.I
        
        self._msave()
        for t in range(params.T):
            self.t = t
            self._sim_step(params)
            self._msave()
            
        return Results(
            N=self.N,
            frac_s_daily=np.array(self.m['S']) / self.N,
            frac_i_daily=np.array(self.m['I']) / self.N,
            frac_r_daily=np.array(self.m['R']) / self.N,
        )
        
    def _sim_step(self, params):
        #print(self.tsex)
        for a,b in self.tsex[self.t]:
            
            # expose new peeps
            if rnd() < params.bern_si: 
                if self.X[a] == self.I and self.X[b] == self.S:
                    self.X[b] = self.I
                if self.X[a] == self.S and self.X[b] == self.I:
                    self.X[a] = self.I
            
        # move from infected
        R = np.random.random( self.N )
        for ii in np.argwhere( (R<params.bern_ir)&(self.X==self.I) ).flatten():
            self.X[ii] = self.R
            #print('recovered', self.t, ii )

class EventNetworkDiscreteTimeSEIR(Model):
    
    S = 0
    E = 1
    I = 2
    R = 3
    
    parameters = [
        'events', # node1, node2, time
        'bern_se',
        'bern_ei',
        'bern_ir',
        'T',
        'frac_i_0',
        'frac_r_0'
    ]
    
    def __init__(self):
        self.m = defaultdict(list)
        self.t = 0
        
        
    def _msave(self):
        self.m['S'].append( np.sum(self.X==self.S) )
        self.m['E'].append( np.sum(self.X==self.E) )
        self.m['I'].append( np.sum(self.X==self.I) )
        self.m['R'].append( np.sum(self.X==self.R) )
        
        # do I need the more detailed state?
        # self.m['X'].append( self.X.copy() )
        
    def simulate(self, params):        
        # rearranging for efficient access
        self.tsex = defaultdict(list)
        self.nodes = set()
        for t,a,b in params.events:
            self.nodes.update({a,b})
            
        self.nodes = sorted(self.nodes)
        self.n2i = {n:i for i,n in enumerate(self.nodes)}
        self.i2n = {i:n for i,n in enumerate(self.nodes)}
        
        # converting these names to mere integers
        for t,a,b in params.events:
            self.tsex[t].append((self.n2i[a],self.n2i[b]))
            
        self.N = len(self.nodes)
        
        self.X = np.array([self.S for _ in range(self.N)])
        
        for i in sample(range(self.N), int(params.frac_i_0*self.N)):
            self.X[i] = self.I
        
        self._msave()
        for t in range(params.T):
            self.t = t
            self._sim_step(params)
            self._msave()
            
        return Results(
            N=self.N,
            frac_s_daily=np.array(self.m['S']) / self.N,
            frac_e_daily=np.array(self.m['E']) / self.N,
            frac_i_daily=np.array(self.m['I']) / self.N,
            frac_r_daily=np.array(self.m['R']) / self.N,
        )
        
    def _sim_step(self, params):
        #print(self.tsex)
        for a,b in self.tsex[self.t]:
            
            # expose new peeps
            if rnd() < params.bern_se: 
                if self.X[a] == self.I and self.X[b] == self.S:
                    self.X[b] = self.E
                if self.X[a] == self.S and self.X[b] == self.I:
                    self.X[a] = self.E
            
        # move from incubation
        R = np.random.random( self.N )
        for ii in np.argwhere( (R<params.bern_ei)&(self.X==self.E) ).flatten():
            self.X[ii] = self.I

        # move from infected
        R = np.random.random( self.N )
        for ii in np.argwhere( (R<params.bern_ir)&(self.X==self.I) ).flatten():
            self.X[ii] = self.R
            #print('recovered', self.t, ii )
            
# borrowing liberally from Epidemics on Networks...
            
import EoN
class ContinuousTimeSIR:
    def simulate(self, params, **kwargs):
        args = dict(
            G=params.net,
            tau=params.beta_si,
            gamma=params.beta_ir,
            tmax=params.T,
            **kwargs # new idea for return_full_data
        )
        
        if 'frac_i_0' in params:
            args['rho'] = params.frac_i_0
            
        elif 'nodes_i_0' in params:
            args['initial_infecteds'] = params.nodes_i_0
            
        #t, S, I, R = EoN.fast_SIR(**args)
        
        return EoN.fast_SIR(**args)

class ContinuousTimeSEIR:
    def simulate(self, params):
        
        self.N = len(params.net.nodes)

        H = nx.DiGraph()
        H.add_node('S')
        H.add_edge('E', 'I', rate = params.beta_ei)
        H.add_edge('I', 'R', rate = params.beta_ir)

        J = nx.DiGraph()
        J.add_edge(('I', 'S'), ('I', 'E'), rate = params.beta_se)
        IC = defaultdict(lambda: 'S')
        
        if 'frac_i_0' in params:
            for node in sample(list(params.net.nodes), int(self.N * params.frac_i_0)):
                IC[node] = 'I'
        elif 'nodes_i_0' in params:
            for node in params.nodes_i_0:
                IC[node] = 'I'
        else:
            raise Exception('specify either nodes_i_0 or frac_i_0')

        return_statuses = ('S', 'E', 'I', 'R')

        t, S, E, I, R = EoN.Gillespie_simple_contagion(params.net, H, J, IC, return_statuses,
                                                tmax = float('Inf'))

        return t,np.array(S),np.array(E),np.array(I),np.array(R)

class ContinuousTimeSEIRS:
    def simulate(self, params, return_full_data=False):
        
        self.N = len(params.net.nodes)

        H = nx.DiGraph()
        H.add_node('S')
        H.add_edge('E', 'I', rate = params.beta_ei)
        H.add_edge('I', 'R', rate = params.beta_ir)
        H.add_edge('R', 'S', rate = params.beta_rs)
        
        J = nx.DiGraph()
        J.add_edge(('I', 'S'), ('I', 'E'), rate = params.beta_se)
        
        IC = defaultdict(lambda: 'S')
        for node in sample(list(params.net.nodes), int(self.N * params.frac_i_0)):
            IC[node] = 'I'

        return_statuses = ('S', 'E', 'I', 'R')

        if not return_full_data:
            t, S, E, I, R = EoN.Gillespie_simple_contagion(params.net, H, J, IC, return_statuses,
                                                    tmax = float('Inf'))

            return t,np.array(S),np.array(E),np.array(I),np.array(R)
        else:
            return EoN.Gillespie_simple_contagion(params.net, H, J, IC, return_statuses,
                                                    tmax = params.T, return_full_data=True)
    
    
class DiscreteTimeSIR:
    def simulate(self, params):
        
        self.N = len(params.net.nodes)
        t, S, I, R = EoN.basic_discrete_SIR(params.net, params.bern_p, rho=params.frac_i_0)
        return t,np.array(S),np.array(I),np.array(R)