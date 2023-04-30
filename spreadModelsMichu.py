import sys
sys.path.append('/Users/alecmcgail/Documents/GitHub/mitchu')

from mitchu import *
from mitchu.lib import *
from mitchu.metrics import *

class infection_from_outside(Action):
    def __init__(self, lam=0.01):
        self.lam = lam
        
        super().__init__()
    
    def act_time(self):
        if self.person.state != 'S':
            return float('inf')
        
        return RV('exp', 1/self.lam)
    
    def act(self, context):
        if self.person.state == 'S':
            context.log(" - ".join(['infecting from outside', str(self.person), f"{context.CURRENT_WORLD_TIME:0.1f} days"]))
            self.person.state = 'I'

class infection_from_friend(Action):
    def __init__(self, beta=0.3):
        self.beta = beta
        
        super().__init__()
    
    def act_time(self):
        n = sum(p.state == 'I' for p in self.person.contacts)
        
        if self.person.state != 'S' or n == 0:
            return float('inf')
        
        return RV('exp', self.beta * n)
    
    def act(self, context):
        if self.person.state == 'S':
            context.log(" - ".join(['infecting from friend', str(self.person), f"{context.CURRENT_WORLD_TIME:0.1f} days"]))
            self.person.state = 'I'

class recovery(Action):
    def __init__(self, beta=0.3):
        self.beta = beta
        
        super().__init__()
    
    def act_time(self):
        if self.person.state != 'I':
            return float('inf')
        
        return RV('exp', self.beta)
    
    def act(self, context):
        self.person.state = 'R'
        

class StateMetric(Metric):
    def measure(self):
        return sum( p.state == self.state for p in self.context.ppl )
    
    def show(self, tstart=0, tstop=100, **kwrargs):        
        alltimes = sorted(self.snaps.keys())
        alltimes = [x for x in alltimes if tstart<=x<=tstop]
        
        dd = pd.DataFrame(dict({
            self.state: [self.snaps[ t ] for t in alltimes]
        }, t=alltimes))

        plt.plot(dd.t, dd[self.state], **kwrargs) # , legend=None
        plt.xlabel("time")
    
        
class S(StateMetric):
    state='S'
class E(StateMetric):
    state='E'
class I(StateMetric):
    state='I'
class R(StateMetric):
    state='R'