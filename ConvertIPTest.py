import geocoder
import numpy as np
import sklearn as sl
import pandas as pd
import random


def int_to_dqn(st):
    """
    Convert integer to dotted quad notation
    """
    st = "%08x" % (st)
    ###
    # The same issue as for `dqn_to_int()`
    ###
    return "%i.%i.%i.%i" % (int(st[0:2], 16), int(st[2:4], 16), int(st[4:6], 16), int(st[6:8], 16))

df = {'ip': [47456, 77065, 27849]}
data = pd.DataFrame(df)
data['ip'] = data['ip'].apply(lambda x: int_to_dqn(x))
data['coordinate'] = data['ip'].apply(lambda x: geocoder.ip(x).latlng)
print(data)