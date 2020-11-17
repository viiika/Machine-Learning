#单独处理week1
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df1 = pd.read_csv('week1.csv')
# df1.drop('week',axis=1,inplace=True)
#
# id_road = df1['id_road'].unique()
# df2=df1

#6->3

#+ train_TTI_6plus3.csv 分为周一到周日7份表

import datetime
df5 = pd.read_csv('train_TTI_6plus3.csv')
df5.set_index(['id_road', 'time'], inplace=True)
df5['week'] = df5.apply(lambda x: datetime.datetime.strptime(x['date'],"%Y-%m-%d").weekday(),axis=1)

for i in range(7):
    tmp=df5[df5['week']==i]
    tmp.to_csv('train_TTI_6plus3_week'+str(i+1)+'.csv')
