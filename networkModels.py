from .meta import *
from .common import *

class NetworkedBernoilliEvents(Model):
    
    defaults = Params(
        bern_p_event = 0.1, # probability of event in a given day
    )
    
    def simulate(self, params):
        # simply roll a die for each day, for each tie
        
        ties = list(params.net.edges)        
        events = []
        
        roll = np.random.random( len(ties) * params.T )
        events += [ 
            (t,a,b)
            for R,(a,b,t) in zip(roll,[ (a,b,t)
                for a,b in ties
                for t in range(params.T)
            ])
            if R < params.bern_p_event
        ]
        
        return Results(
            events=events
        )

class CompleteNet(Model):
    def simulate(self, params):
        G = nx.complete_graph(params.N)
        return Results(
            net=G
        )
    
class PlaneLatticeNet(Model):
    def simulate(self):
        w = int(np.sqrt(params.N))
        G = nx.grid_2d_graph(w, w, periodic=False)
        
        return Results(
            net=G,
            N= w
        )
    
class ErdosRenyiNet(Model):
    def simulate(self, params):
        G = nx.fast_gnp_random_graph(params.N, params.erdosrenyi_p)
        
        return Results(
            net=G
        )
    
class SmallWorldNet(Model):
    def simulate(self, params):
        G = nx.watts_strogatz_graph(params.N, params.smallworld_k, params.smallworld_p)
        
        return Results(
            net=G
        )
    
class PreferentialNet(Model):
    def simulate(self, params):
        G = nx.barabasi_albert_graph(params.N, params.Edegree)
        
        return Results(
            net=G
        )
    
class PreferentialTriangleNet(Model):
    """
    The average clustering has a hard time getting above a certain cutoff that depends on m. This cutoff is often quite low. The transitivity (fraction of triangles to possible triangles) seems to decrease with network size.
    It is essentially the Barabási–Albert (BA) growth model with an extra step that each random edge is followed by a chance of making an edge to one of its neighbors too (and thus a triangle).
    This algorithm improves on BA in the sense that it enables a higher average clustering to be attained if desired.
    It seems possible to have a disconnected graph with this algorithm since the initial m nodes may not be all linked to a new node on the first iteration like the BA model.
    """
    def simulate(self, params):
        G = nx.powerlaw_cluster_graph(params.N, params.Edegree0, params.triangle_p)
        
        return Results(
            net=G
        )
    
# community graphs make sense
# https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.community



class ConnectedNets(Model):
    
    def simulate(self, params):
        # start by generating new names, based on their group
        # and then add between-group ties
        
        params.micro_nets = list(params.micro_nets)
        nnets = len(params.micro_nets)
        
        names = [f"{i}.{j}" for i,net in enumerate(params.micro_nets) for j in net.nodes]
        edges = [(f"{i}.{j}", f"{i}.{k}") for i,net in enumerate(params.micro_nets) for j,k in net.edges]
        
        assert( type(params.macro_net) == np.ndarray )
        assert( params.macro_net.shape == (nnets, nnets) )
        assert( np.all( params.macro_net.T == params.macro_net ) ) # symmetric
        
        for i,net1 in enumerate(params.micro_nets):
            for j,net2 in enumerate(params.micro_nets):
                if i>=j:continue
                
                poss_edges = [
                    (n1,n2)
                    for n1 in net1.nodes
                    for n2 in net2.nodes
                    if n1 < n2
                ]
                
                rolls = np.random.random( len(poss_edges) )
                new_edges = [
                    (f"{i}.{a}", f"{j}.{b}") for (a,b),R in zip(poss_edges, rolls)
                    if R < params.macro_net[i,j]
                ]
                #print('added', len(new_edges), 'edges')
                #print(edges[:10], new_edges[:10])
                edges += new_edges
              
        net = nx.Graph()
        net.add_nodes_from( names )
        net.add_edges_from( edges )
        
        return Results(
            net = net
        )

class FarzNet(Model):
    """
    group_size_dist, or `phi` -- 1 is power law, higher is more balanced
    max_groups_per_person, or `r` -- default 1 results in disjoint communities
    p_multiple_communities, or `q` -- default is 0.5
    p_connect_neighbors, or `t`, default 0
    common_neighbors_effect, `alpha`, default 0.5
    degree_similarity_effect, `gamma`, default 0.5
    p_edge_communities, `beta` -- default 0.8
    p_random_edges, `epsilon` -- default is 0.0000001
    """
    defaults = Params(
        group_size_dist=1,
        max_groups_per_person=2,
        p_multiple_communities=0.5,
        p_connect_neighbors=0,
        common_neighbors_effect=0.5,
        degree_similarity_effect=0.5,
        p_edge_communities=0.8,
        p_random_edges=0.0000001,
        n_communities=4,
        N=1000,
        Edeg0=2
    )
    
    def simulate(self, params):
        params = self.defaults.extend(params)
        from FARZ import Graph,Comms,select_node,connect,assign

        G =  Graph()
        C = Comms(params.n_communities)
        for i in range(params.N):
        #         if i%10==0: print '-- ',G.n, len(G.edge_list)
            G.add_node()
            assign(i, C, params.group_size_dist, params.max_groups_per_person, params.p_multiple_communities)
            connect(i, params.p_connect_neighbors, G, C, params.common_neighbors_effect, params.p_edge_communities, params.degree_similarity_effect, params.p_random_edges)
            for e in range(1,params.Edeg0//2):
                j = select_node(G) 
                connect(j, params.p_connect_neighbors, G, C, params.common_neighbors_effect, params.p_edge_communities, params.degree_similarity_effect, params.p_random_edges)        

        import networkx as nx
        G = G.to_nx(C)
        
        return Results(
            net=G
        )
    
    

def pair_stubs(stubs, likelihood=None):
    if likelihood is None:
        likelihood = 1 - np.identity(max(stubs)+1)

    stubs = list(stubs)

    pairs = []
    while len(stubs):
        # choose a random stub
        a = choice(stubs)

        # based on likelihood, choose an alter
        bs = likelihood[a]
        ix = [x in stubs and x!=a for x in range(len(bs))]

        #print(bs)
        #print(ix)
        denom = bs[ix].sum()
        if denom == 0: # there's no one I can even connect to...
            stubs.remove(a)
            print('death')
            continue

        b = np.random.choice( np.array(range(len(bs)))[ix], size=1, p=bs[ix]/denom )[0]

        stubs.remove(a)
        stubs.remove(b)

        pairs.append((a,b))

    return pairs

def make_degree_distribution(N, dd=None, degrees=None, likelihood=None):
    if degrees is None:
        ddtot = sum(dd.values())
        k = sorted(dd.keys())

        p = np.array([dd[kk] for kk in k])
        p /= p.sum()

        degrees = np.random.choice(k, size=N, p=p)

    stubs = [i for i,d in enumerate(degrees) for _ in range(d)]

    edges = pair_stubs(stubs, likelihood)
    return edges

    
class ConfigurationNet:
    
    defaults = Params(
        N = 100,
        degree_distribution = None,
        degrees = None,
        likelihood = None
    )
    
    def simulate(self, params):
        params = self.defaults.extend(params)
        
        edges = make_degree_distribution( params.N, dd=params.degree_distribution, degrees=params.degrees, likelihood=params.likelihood )
        G = nx.Graph()
        G.add_nodes_from(range(params.N))
        G.add_edges_from(edges)
        return Results(
            net=G
        )
    

class FociNetwork:
    def __init__(self, N, p_paired, n_schools, n_workplaces, n_grocery):
        self.N = N
        self.p_paired = p_paired
        self.n_schools = n_schools
        self.n_workplaces = n_workplaces
        self.n_grocery = n_grocery
        
        self.families = []
        self.schools = defaultdict(list)
        self.workplaces = defaultdict(list)
        self.grocery = defaultdict(list)
        
        self.school_children = set()
        self.children = set()
        
        self._build_families()
        self._build_schools()
        self._build_workplaces()
        self._build_grocery()
        
        self.net = defaultdict(int)
        self._build_net()
        self._build_flattened_net()
        
    def _build_families(self):
        # pair people up
        last_paired = None
        
        i = 0
        while i < self.N:
            if random() < self.p_paired:        
                if last_paired is not None:
                    fam = [last_paired, i]
                    last_paired = None

                    n_children = randint(0,3)
                    for _ in range(n_children):
                        i += 1
                        if i >= self.N:break
                        
                        fam.append(i)
                        self.children.add(i)

                    self.families.append(fam)
                else:
                    last_paired = i

            i += 1
            
    def _build_schools(self):
        # choose one person from each household to go to one of 30 schools
        for f in self.families:
            pc = np.array([x in self.children for x in f])
            pc = pc/pc.sum()
            who = np.random.choice(f)

            self.schools[ randint(1, self.n_schools) ].append(who)
            self.school_children.add(who)
            
    def _build_workplaces(self):
        # build workplaces through preferential attachment
        for f in self.families:
            pc = np.array([x not in self.school_children for x in f])
            pc = pc/pc.sum()
            who = np.random.choice(f, p=pc)

            which_place_p = np.array([ len(self.workplaces[i]) + 1 for i in range(self.n_workplaces) ])
            which_place_p = which_place_p / which_place_p.sum()

            which_place = np.random.choice( range(self.n_workplaces), p=which_place_p )

            self.workplaces[ which_place ].append(who)
            
    def _build_grocery(self):
        # choose one person from each household to go to one of the grocery stores
        for f in self.families:
            pc = np.array([x not in self.school_children for x in f])
            pc = pc/pc.sum()
            who = np.random.choice(f, p=pc)

            self.grocery[ randint(1, self.n_grocery) ].append(who)

    def _build_net(self):
        gsets = [
            self.families,
            self.schools.values(),
            self.workplaces.values(),
            self.grocery.values()
        ]

        for gg in gsets:
            for g in gg:

                for i in g:
                    for j in g:
                        if i>=j:continue

                        self.net[i,j] += 1/len(g)

    def _build_flattened_net(self):
        nstar = nx.Graph()
        nstar.add_weighted_edges_from( [(a,b,c) for (a,b),c in self.net.items()] )
        self.flattened = nstar

    def get_2mode_net(self):
        gsets = [
            self.families,
            self.schools.values(),
            self.workplaces.values(),
            self.grocery.values()
        ]

        gset_names = ['families', 'school', 'workplace', 'grocery']
        edges = []

        for ggi, gg in enumerate(gsets):
            for gi, g in enumerate(gg):
                mygname = gset_names[ggi] + "-" + str(gi)

                for i in g:
                    edges.append( (i, mygname) )

        nstar = nx.Graph()
        nstar.add_edges_from(edges)
        return nstar

    def simulate(self, **args):
        return ContinuousTimeSIR().simulate(Params(
            net = self.flattened,
            **args
        ), return_full_data=True, transmission_weight='weight')


# create a class which extends FociNetwork
class AddAccumulativeAdvantage(FociNetwork):
    def __init__(self, base, alpha=0.5, beta=1, target_degree=3):
        self.base = base
        self.base_net = base.flattened

        self.alpha = alpha
        self.beta = beta
        self.target_degree = target_degree

        self.net = defaultdict(int, self.base.net)

        self._build_net()
        self._build_flattened_net()


    def _build_net(self):
        # weighted degree distribution
        all_weights = []
        deg = defaultdict(int)
        for (a,b,w) in self.base_net.edges.data('weight'):
            if a > b: continue
            deg[a] += w
            deg[b] += w
            all_weights.append(w)

        # weight of accumulative advantage
        alpha = self.alpha
        # base weight
        beta = self.beta

        nodes = sorted(deg)

        d_ = np.array([deg[x] for x in nodes])
        current_degree = np.mean(d_)

        if current_degree > self.target_degree:
            raise ValueError("Current degree is already greater than target degree")

        avg_weight = np.mean(all_weights)
        degree_diff = self.target_degree - current_degree
        to_go = degree_diff/(avg_weight / 10000)

        BATCH_SIZE = int(to_go // 100)

        # rewrite to choose edges N individuals
        # then choose N individuals to be attached to, weighted by degree by alpha

        deg_snapshots = []
        while current_degree < self.target_degree:
            # pick random weights
            w = np.random.choice(all_weights, size=BATCH_SIZE, replace=True)

            # choose random nodes to be attached to a new person
            a = np.random.choice(nodes, size=BATCH_SIZE, replace=True)

            # choose random alters, weighted by degree by alpha
            p = np.power(d_, alpha) + beta
            p = p/p.sum()

            b = np.random.choice(nodes, size=BATCH_SIZE, p=p, replace=True)

            # add the new weighted edges
            for i in range(BATCH_SIZE):
                x,y = a[i],b[i]
                if x > y: x,y = y,x

                if x == y: continue # don't add self-loops
                self.net[(x,y)] += w[i]
                
            # update degree
            for i in range(BATCH_SIZE):
                if a[i] == b[i]: continue # don't add self-loops
                deg[a[i]] += w[i]
                deg[b[i]] += w[i]

            # update current degree
            current_degree = np.mean(list(deg.values()))

            # save snapshot
            deg_snapshots.append(current_degree)

    def plot_wdegree_distribution(self):
        n = self.flattened
        degrees = [n.degree(n_, weight='weight') for n_ in n.nodes()]
        plt.figure(figsize=(3,1))
        plt.hist(degrees);

    def plot_degree_distribution(self):
        n = self.flattened
        degrees = [n.degree(n_) for n_ in n.nodes()]
        plt.figure(figsize=(3,1))
        plt.hist(degrees);