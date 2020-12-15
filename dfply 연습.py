from dfply import *
import pandas as pd
import os
import numpy as np
import matplotlib 
import matplotlib.dates as md
import matplotlib.pyplot as  plt
import seaborn as sns

os.getcwd()
df = pd.read_csv('csv/의료인_2차.csv', encoding='utf-8')
df.head()

df['burnout']
list(df.columns)
df2 = df >> select(X.pdi) >> drop()
df2 >> group_by(X.Occupation) >> summarize(mean_bo = X.burnout.sum(), mean_pdi = X.pdi.mean())

df2 >> select(0,['occu_a','burnout'])
df >> select(starts_with('bo'))
df >> select(~starts_with('pdi'))
df2 >> drop(columns_from('pdi')) ## columns_from(): 선택을 시작하고자 하는 변수
df2 >> select(columns_to(0, inclusive=True), 'pdi', columns_from(-1)) # columns_to(): 열번호로 선택하고자 하는 열 선택
df >> filter_by(X.burnout>3.2) >> select(contains("pdi"))

df >> select(columns_to('pdi_13'),columns_from('pdi_1'))
df2
df2 >> arrange(X.burnout, ascending = False)
df2 >> group_by(X.Occupation) >> summarise_each([np.mean, np.std, np.min, np.max], X.burnout, X.pdi)
df2 >> plt.plot(x="Occupation", y="mean_pdi",kind="barh")
df2 >>= rename(Occupation = X.occu_a)
df2.columns


# plot 환경설정
plt.style.use('seaborn')
ax = plt.subplots(1,1)

import os
os.getcwd()
import pandas as pd
import numpy as np

crime = pd.read_csv('data/crime_seoul.csv', encoding='euc-kr')
crime.head()

df = pd.read_csv('C:\\Users\\82105\\Desktop\\Python\csv\\의료인_2차.csv', encoding = 'utf-8')
df['Q5_n3'].values
data = df >> dfply.filter_by(df.Q5_n3 == 2)

