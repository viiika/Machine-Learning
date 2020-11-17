# 各路每周TTI可视化
import time
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('train_TTI.csv')
df1['time'] = pd.to_datetime(df1['time'])
# df1

df12=df1
df12['week'] = df1.apply(lambda x: x['time'].weekday(),axis=1)
df12.to_csv('df12.csv')

weeklist=[]#按周一到周日存为7个dateframe
for i in range(7):
    tmp=df12[df12['week']==i]

    # 处理每周时间缺失
    df3 = pd.DataFrame()
    period1 = pd.date_range('2019-01-01 00:00:00', '2019-03-31 23:50:00', freq='10T')
    period2 = pd.date_range('2019-10-01 00:00:00', '2019-12-21 23:50:00', freq='10T')
    period = period1.append(period2)
    for name, group in tmp.groupby('id_road'):
        group.set_index('time', inplace=True)
        group = group.reindex(period, method='nearest')
        group['time'] = group.index
        group.reset_index(drop=True, inplace=True)
        df3 = df3.append(group)

    tmp.to_csv('week'+str(i+1)+'.csv',index=False)
    weeklist.append(df3)

# 以下是绘图部分
curweek = 0
namelist = []
for week in weeklist:
    for name, group in week.groupby(week['id_road']):
        namelist.append(name)
# namelist

# #按星期几绘图
# for week in weeklist:
#     for name, group in week.groupby(week['id_road']):
#         plt.figure()
#         group.set_index('time', inplace=True)
#         group.index = group.index.time
#         group['TTI'].groupby(group.index).mean().plot()
#         plt.title(str(name)+"-"+str(curweek))
#     curweek+=1
#     break
#     #太多了一次画不出来

# #按哪个路线绘图
for week in weeklist:
    for name, group in week.groupby(week['id_road']):
        if name != 276183:
            continue  # 太多了一次画不出来
        plt.figure()
        group.set_index('time', inplace=True)
        group.index = group.index.time
        group['TTI'].groupby(group.index).mean().plot()
        plt.title(str(name) + "-" + str(curweek))
    curweek += 1
