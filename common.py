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