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