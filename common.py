import numpy as np
import networkx as nx

from collections import defaultdict, Counter

from matplotlib import pyplot as plt

from random import randint, random, choice, sample, shuffle
from random import random as rnd

from itertools import groupby

from pathlib import Path

from pathlib import Path
from csv import reader
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from collections import Counter
from random import random
from collections import defaultdict

from random import sample, choice
import datetime as dt
from IPython.display import Markdown, HTML, display

import json

import covidcast
from datetime import date

def county_map(dta, locations, colors):
    import plotly.express as px
    rng = tuple(x(dta[colors]) for x in [min, max])

    fig = px.choropleth_mapbox(dta, geojson=counties, locations=locations, color=colors,
                               color_continuous_scale="Viridis",
                               range_color=rng,
                               mapbox_style="carto-positron",
                               zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                               opacity=0.5,
                               labels={'value':'val'}
                              )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()



class CovidCastCache:
    def __init__( self, d='covidcast' ):
        self.d = d
        
        Path(d).mkdir(exist_ok=True)
        
    # gets and persists to disk
    def get_day(self, space, indicator, d, level="county"):
        parts = [space,indicator]
        parts += [d.strftime('%Y-%m-%d')+".csv"]
        fn = ", ".join(parts)
        
        fold = Path(self.d, level, space, indicator)
        fold.mkdir(exist_ok=True, parents=True)
        fn_path = fold / fn
        
        if fn_path.exists():
            return pd.read_csv(fn_path)
        else:
            res = covidcast.signal(space, indicator, d, d, level)
            res.to_csv( fn_path )
            return res
        
    def get(self, space, indicators, date_start, date_end, level="county"):
        if type(indicators) == str:
            indicators = [indicators]
            
        ndays = (date_end-date_start).days + 1
        
        dfs = []
        
        for indicator in indicators:
            parts = []
            for d in [ date_start+dt.timedelta(days=i) for i in range(ndays) ]:
                myday = self.get_day(space, indicator, d, level)
                
                myday.time_value = pd.to_datetime( myday.time_value )
                myday.geo_value = myday.geo_value.map( lambda x:str(x).zfill(5) )
                
                parts.append( myday )

            df = pd.concat(parts)
            df = df.set_index(['geo_value','time_value'])
            dfs.append(df)
            
        return dfs
    
    def summary(self):
        
        for level in Path(self.d).glob("*"):
            if not level.is_dir():continue
            
            print(level.name)

            for space in Path(level).glob("*"):
                if not space.is_dir():continue
                
                print(f" {space.name}")

                for indicator in Path(space).glob("*"):
                    if not indicator.is_dir():continue
                    
                    n = len(list(indicator.glob("*.csv")))
                    
                    print(f"  {indicator.name} - {n} items")
                    