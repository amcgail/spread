from .common import *

from .meta import *
from .networkModels import *
from .spreadModels import *

def to_discrete(ts, xs, tmax):
    tf = np.floor( ts )
    discrete_flips = np.argwhere( tf[:-1] != tf[1:] ).flatten()

    lastt = 0
    lasti = 0
    vals = []
    for nxt in discrete_flips:
        now_time = int(ts[nxt])+1
        for tt in range(lastt, min(tmax+1,now_time)):
            vals.append( xs[lasti] )   
            #print(tt, xs[lasti], 'switch at', ts[nxt])

        lastt = now_time
        lasti = nxt

    for tt in range(lastt, tmax+1):
        vals.append( xs[lasti] )
        
    return vals

# setting getting environment variables for python package...
import yaml

class Params:
    def __init__(self, filename=None):
        self._params = {}
        self.filename = filename
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        if not Path(filename).exists():
            self._params = {}
        else:
            with open(filename, 'r') as f:
                self._params = yaml.safe_load(f)

                if self._params is None:
                    self._params = {}

    def save(self, filename=None):
        if filename is None:
            filename = self.filename
        with open(filename, 'w') as f:
            yaml.dump(self._params, f)

    def get(self, key, default=None):
        return self._params.get(key, default)

    def set(self, key, value):
        self._params[key] = value

    def __getattr__(self, key):
        if key not in self._params:
            raise AttributeError(f"Parameter '{key}' not found")
        return self._params[key]

    def __setattr__(self, key, value):
        if key in {'_params', 'filename'}:
            super().__setattr__(key, value)
        else:
            self._params[key] = value

    def __repr__(self):
        return f"Params({self._params})"

params = Params('.config.yaml')