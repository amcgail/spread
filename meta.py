
class Model:
    pass

class Params:
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            d = args[0]
        elif len(args) == 0:
            d = kwargs
        else:
            raise Exception('what are your args again?')
            
        self.d = d
        
    def __getattr__(self, k):
        return self.d[k]
    
    def extend(self, other):
        return Params( dict(self.d, **other.d) )

    def keys(self):
        return self.d.keys()
    
    def values(self):
        return self.d.values()
    
    def items(self):
        return self.d.items()

    def __getitem__(self, k):
        return self.d[k]

    def __setitem__(self, k, v):
        self.d[k] = v

    def __delitem__(self, k):
        del self.d[k]

    def __iter__(self):
        return iter(self.d)

    def __len__(self):
        return len(self.d)
    
    def __contains__(self, what):
        return what in self.d
    
    def __repr__(self):
        return "<Params: " + ", ".join( f"{a}={b}" for a,b in self.d.items() ) + ">"

class Results:
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            d = args[0]
        elif len(args) == 0:
            d = kwargs
        else:
            raise Exception('what are your args again?')
            
        self.d = d
        
        # don't know yet what to do with this
        self.conversions = {}
        
    def __getattr__(self, k):
        if k in self.d:
            return self.d[k]
        elif k in self.conversions:
            return self.conversions[k]()
        
    def __repr__(self):
        return "<Results: " + ", ".join( f"{a}" for a,b in self.d.items() ) + ">"

class D:
    pass

class Unif(D):
    def __init__(self, a, b):
        self.a, self.b = a,b
    def gen(self):
        from random import random
        a,b = self.a, self.b
        return random()*(b-a) + a

class Unif(D):
    def __init__(self, a, b):
        self.a, self.b = a,b
    def gen(self):
        from random import random
        a,b = self.a, self.b
        return random()*(b-a) + a

class Choose(D):
    def __init__(self, *args):
        if len(args) == 1:
            self.choices = args[0]
        else:
            self.choices = args
            
    def gen(self):
        from random import choice
        return choice(self.choices)

class Normal(D):
    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu, sigma
    def gen(self):
        from random import gauss
        return gauss(self.mu, self.sigma)

class IntBetween(D):
    def __init__(self, a, b):
        self.a, self.b = a, b
    def gen(self):
        from random import randint
        return randint(self.a, self.b)

class ParamField:
    def __init__(self, *args, **kwargs):
        self.d = kwargs
        
    def gen(self):
        res = {}
        for k,v in self.d.items():
            if issubclass(v.__class__, D):
                res[k] = v.gen()
            else:
                res[k] = v
                
        return Params(**res)
        