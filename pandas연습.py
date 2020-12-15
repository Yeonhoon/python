import pandas as pd
import numpy as np
from pandas import Series, DataFrame
obj = pd.Series([4,7,-5,3])
obj
obj.values
obj.index

obj2 = pd.Series([4,7,-5,3], index=['d','b','a','c'])
obj2.index

obj2['a']
obj2['e']=10
obj2[['c','d','a','e']]

obj2[obj2>2]

'b' in obj2
'f' in obj2

sdata = {'ohio' : 35000, 'Texas' : 71000, 'Oregon' : 16000, 'Michigan':5000}
sdata
obj3 = pd.Series(sdata)
obj3

states=['ohio','Texa','D.C']
obj4 = pd.Series(sdata, index=states)
obj4
pd.isnull(obj4)
pd.notnull(obj4)
obj4.isnull()
obj3 + obj4
obj3

# Series의 장점: 산술 연산에서 색인과 라벨 자동정렬.

#index 이름 붙이기:.index.name
obj4.name = "population"
obj4.index.name = "states"
obj4



## DataFrame

data = {'states':['Ohio','Ohio','Ohio','Nevada','Nevada','Nevada'],
        'year': [2000,2001,2002,2001,2002,2003],
        'pop':[1.5,1.7,3.6,4.2,2.9,3.2]}

frame = pd.DataFrame(data)
frame

#set_index('index'): index 열 정해주기
id = np.arange(6)
id
frame.set_index(id)
frame.index.name = "num"
frame

frame2=pd.DataFrame(data, columns= ['year','states','pop'])
frame2

frame2['debt']= 16.5
frame2['debt']=[1,5,1,4,7,9]
frame2.index=['one','two','three','four','five','six']
frame2.index.name="num"
frame2

#Series.copy()
frame_copy = frame2.copy()
frame_copy

frame2.index.name='num';frame2.columns.name='columns'
frame2

pop = {'Nevada':{2001:2.4, 2002:2.9},
    'Ohio':{2000:2.5, 2001:1.7, 2002:3.6}}

frame3 = pd.DataFrame(pop)
frame3
frame3.T

frame3.values


# pandas의 핵심 기능
# info(): 데이터셋 정보 찾기

tips.info()


obj = pd.Series([4.5,7.2,-5.3,3.6,], index=['d','b','a','c'])

obj2=obj.reindex(['a','b','c','d','e'])
obj2['e']=1
obj2
obj3= pd.Series(['blue','purple','yellow'], index=[1,2,3])
obj3.reindex(range(6), method ="ffill") #reindex: 행, 열 모두 변경 가능.ArithmeticError

import numpy as np
pd.DataFrame(np.arange(16).reshape((4,4)),
            index= ['a','b','c','d'],
            columns = ["Ohio",'Michigan',"Pensylbenia",'Arizona'])


frame.reindex(['a','b','c','d','e','f'])

# 하나의 행이나 열 제거하기. drop
obj = pd.Series(np.arange(5.), index=['a','b','c','d','e'])
obj
new_obj = obj.drop('c')
new_obj

data=pd.DataFrame(np.arange(16).reshape((4,4)),
            index= ['a','b','c','d'],
            columns = ["Ohio",'Michigan',"Pensylbenia",'Arizona'])

data.drop(['Ohio'], axis=1)
data.drop(['Ohio'], inplace=True, axis=1) # inplace=T 하면 drop한 것이 data에 새롭게 반영되어 저장됨.

## 여러 행 제거
data.drop(['a','b,''c'], axis = '')
data.assign()
# 색인: 테이블 안에서 값 찾기
obj['a':'d'] # 행만 됨
obj[['a','b','c','d']]

data
data[['Michigan','Arizona']]
data['Michigan':'Arizona'] #열은 : 안됨
data.loc[:'Michigan']
data.iloc[2,[3,0,1]]

#loc으로 indexing하기
frame
frame.loc[2:,"states":'year']

#iloc: 숫자로 loc 하기
frame.iloc[:, 1:3]


# pivoting
flights =flight.pivot('year','month', 'passengers') # pivot(row, columns, values)

# 결측치 제거
df.dropna()


ser = pd.Series(np.arange(3.))
ser.iloc[1]

# 산술 연산
s1 = pd.Series([7.3,-2.5,3.4,1.5], index= ['a','c','d','e'])
s2 = pd.Series([-2.1,3.6,1.5,4,3.1], index=['a','c','e','f','g'])

s1 + s2 # 겹치는 index가 없을 경우 결측치.

df1 = pd.DataFrame(np.arange(9).reshape((3,3)), columns=list('bcd'), index = ['Ohio','Texas','Colorado'])
df2 = pd.DataFrame(np.arange(12).reshape((4,3)), columns=list('bce'), index = ['Utah','Ohio','Texas','Oregon'])
df1+ df2 # 겹치지 않는 값은 NA

df1 = pd.DataFrame(np.arange(12.).reshape((3,4)), columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4,5)), columns=list('abcde'))
df1
df2
df2.loc[1,'b']= np.nan

df1 + df2

#fill_value*()
df1.add(df2, fill_value=0) # df1과 df2를 더할때, 결측값을 0으로 채워줌.


## 함수 적용과 매핑
## apply
frame = pd.DataFrame(np.arange(12.).reshape((4,3)), columns = list('bde'), index=['Utah', 'Ohio','Texas','Oregon'])

frame

f = lambda x :x.max() - x.min()
frame.apply(f) # 기본은 컬럼 기준으로
frame.apply(lambda x: x.max()**x.min())
frame.apply(f, axis ='columns') #axis=columns으로 설정하면 로우에 대해 수행

format = lambda x:'%.2f' %x #실수값을 문자열 포맷으로
frame.applymap(format)
frame['e'].map(format)

## 정렬과 순위

#index
obj = pd.Series(range(4), index=['d','a','b','c'])
obj.sort_index() #index기준으로 정렬하기
frame.sort_index(axis=1, ascending=False) 

frame.sort_values(by='b',ascending=False) #열의 value 기준으로 정렬

#rank
frame.append(frame.rank(axis=0))

## 중복색인
frame.rename(columns={'b':'a','d':'a'}, inplace=True)
frame.loc[:,'a']

## 기술통계 및 요약.
df = pd.DataFrame([[1.4, np.nan],[7.1,-4.5],
                  [np.nan, np.nan],[0.75,-1.3]],
                  index=['a','b','c','d'],
                  columns = ['one','two'])

df.sum(axis='columns', skipna=False)
df.idxmax()
df.cumsum()
df.describe()

import pandas_datareader.data as web
all_data = {ticker: web.get_data_yahoo(ticker)
            for ticker in ['AAPL','IBM','MSFT','GOOG']}

price = pd.DataFrame({ticker: data['Adj Close']
                    for ticker, data in all_data.items()})

volume = pd.DataFrame({ticker: data['Volume']
                    for ticker, data in all_data.items()})

returns = price.pct_change()
returns.tail()

# corr: 상관관계
returns['MSFT'].corr(returns['IBM']).round(2)
returns.MSFT.corr(returns.IBM).round(2)
returns.corr() # 전체
returns.corrwith(returns.IBM) # 열 별로 선택해서 상관관계

# cov: 공분산
returns['MSFT'].cov(returns['IBM']).round(2)

obj = pd.Series(['c','a','d','a','a','b','b','c','c'])
uniques =obj.unique()
obj.value_counts(sort=True)

data = pd.DataFrame({'Q1':[1,3,4,3,4],
                    'Q2':[2,3,1,2,3],
                    'Q3':[1,5,2,4,4]})

data.apply(pd.value_counts) # 1~5까지 각 열에서 몇개가 있는지 카운트. 1~5가 없는 경우 NA
data.apply(pd.value_counts).fillna(0) # fillna(): 괄호 안의 값으로 na 채우기


## 데이터 불러오기

df=pd.DataFrame(np.arange(12).reshape((3,4)), index = ['hello','world','foo'],columns=list('abcd'))

import os
os.getcwd()
df.to_csv('df.csv')
cat df.csv

df.iloc[1,1:3]=np.nan
df.fillna(0)

#  csv 파일 불러올 수 없는 경우 수동으로  value 가져오기
import csv
f= open('df.csv')
reader = csv.reader(f)

for line in reader:
    print(line)

with open('df.csv') as f:
    line= list(csv.reader(f))

line
header, values = line[0], line[1:]
header
values

## JSON
import json
obj ="""
    {"name" : "Wes",
    "placed_lived" : ["United States","Spain","Germany"],
    "pet": null,
    "siblings": [{ "name" : "Scott", "age" : 30, "pets" : ["Zeus","Zuko"]},
                {"name" : "Katie", "age" : 38, "pets" : ["Sixes","Stache","Cisco"]}
                ]}
"""

result = json.loads(obj)
type(result)

#json to python 객체
asjson = json.dumps(result)
type(asjson)

siblings = pd.DataFrame(result['siblings'], columns = ['name','age'])
siblings

tables = pd.read_html('python_book/examples/fdic_failed_bank_list.html')
len(tables)
type(tables) # list
failure = tables[0] 
type(failure) # DataFrame
failure.head()
failure.columns
close_timestamps = pd.to_datetime(failure['Closing Date'])
close_timestamps.dt.year.value_counts()


## LXML

from lxml import objectify

path = 'python_book/datasets/mta_perf/Performance_MNR.xml'
parsed = objectify.parse(open(path))
root = parsed.getroot()


## request
import requests
url = "https://api.github.com/repos/pandas-dev/pandas/issues"
resp = requests.get(url)
resp

data = resp.json()
data[0]['title']

# 관심있는 것만 따로 df로 뽑아내기
issues = pd.DataFrame(data, columns = ['number','title','labels','state'])
issues.loc[:,'title']

# python에서 sql 사용하기
import sqlite3

query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
c REAL, d INTEGER);"""

con = sqlite3.connect('mydata.sqlite')
con.execute(query)
con.commit()

data = [('Atlanta','Georgia',1.25,6),
        ('Tallahassee','Florida', 2.6,3),
        ('Sacramento', 'California', 1.7,5)]

stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"

con.executemany(stmt, data)
cursor = con.execute('select * from test')
rows = cursor.fetchall()
cursor.description
pd.DataFrame(rows, columns = [x[0] for x in cursor.description])
# 데이터베이스에 쿼리 보내기는 너무 귀찮은 일임.

import sqlalchemy as sqla
db = sqla.create_engine('sqlite:///mydata.sqlite')
pd.read_sql('select * from test', db)


## 데이터 정제

#1. 누락된 데이터 처리
# isnull(), notnull(), fillna(), dropna()
string_data = pd.Series(['aardvark','artichoke',np.nan, 'avocado'])
string_data.isnull()

string_data[0] = None
string_data.isnull()
string_data.dropna()

from numpy import nan as NA
data = pd.DataFrame([[1., 6.5,3], [1., NA, NA], [NA,NA,NA],[NA,6.5,3.]])

# dropna()
data.dropna(how='all') # row 모두가 na 인 경우만 제거하기
data.dropna(axis=1, how='all')


df = pd.DataFrame(np.random.randn(7,3))
df
df.iloc[:4,1]=NA
df.iloc[:2, 2] = NA

df.dropna(th)

# fillna()
df.fillna({1:0.5, 2:1}) # fillna()에서 사전값{}을 넘겨서 컬럼마다 다른 값으로 채우기 가능

df.iloc[2:,1] = NA
df.iloc[4:,2] = NA
df.fillna(method='ffill', limit=1) # method='ffil' 아래 값으로 다 처리 ,limit 통해 적용하고자 하는 열 선택

df.fillna(df.mean()) #fillna에서 mean() 등의 평균 값을 채워넣을 수 있음.

## 데이터 중복 확인:.duplicated 

data = pd.DataFrame({'k1':['one','two']*3 + ['two'],
                    'k2': [1,1,2,3,3,4,4]})

# drop_duplicates
data.drop_duplicates()

data['v1'] = range(7)
data

# 함수, 매핑을 이용한 데이터 변형
data = pd.DataFrame({'food': ['bacon','pulled pork','bacon','pastrami','corned beef','bacon',
                    'pastrami','honey ham','nova lox'],
                    'ounces': [4,3,12,6,7.5,8,3,5,6]})

data

meat_to_animal = {
    'bacon' : 'pig',
    'pulled pork': 'pig',
    'pastrami': 'cow',
    'corned beef': ' cow',
    'honey ham': 'pig',
    'nova lox': 'salmon'
}

lowercased = data['food'].str.lower()

# map
data['animal']= lowercased.map(meat_to_animal)
data

for i in range(data['ounces']):
    rank = []
    if i < 6:
        rank.append('bad')
    else:
        rank.append('good')
    print(rank)
    data['rank']= pd.DataFrame(rank)
data

data['rank'] = np.where(data['ounces'] < 6, '0' ,'1')

###### 데이터 치환 #replace

data = pd.Series([1., -999., 2., -999., -1000., 3.])
data.replace(-999,NA) # 하나의 값만
data.replace([-999,-1000], NA) # 여러개의 값 동시에 
data.replace([-999,-1000], [NA,0])
data.replace({-999: NA, -1000:0})

idx = ['Ohio','Colorado','New York']
data = pd.DataFrame(np.arange(12).reshape((3,4)),
                    index = idx, columns = list('abcd'))
# rename 으로 columns, index 변경
data.rename(columns = str.upper)
data.rename(index = {'Ohio':'Texas'}, columns = {'a':'z'})

## 개별화와 양자화: cut
ages = [20,22,25,27,21,23,37,31,61,45,41,32]
bins = [18,25,35,60,100]

cats = pd.cut(ages, bins, right =False) # 중괄호 대신 대괄호 위치 변경 가능
cats
data
bins = [4,8,10]
data.values

pd.value_counts(cats) # 범주 세기
group_names = ['Youth','YoungAdult','MiddleAged','Senior']
pd.cut(ages, bins, labels = group_names).value_counts() # 그룹 이름 붙여주기

#qcut: 표준 변위치 사용하여 그룹 나누기
data = np.random.randn(1000)
cats = pd.qcut(data,4) # 4개의 그룹으로 동등한 수 나누기.
cats.value_counts()


## 특이값 찾고 제외하기(데이터 평활화 smoothing)

data = pd.DataFrame(np.random.randn(1000,4)) 
data.describe()

data[2][np.abs(data[2])>3]

# np.sign() 양수냐 음수냐에 따라 1,-1 반환
np.sign(data).head()


# 치환 & 임의 샘플링
df = pd.DataFrame(np.arange(5*4).reshape((5,4)))
df

sampler = np.random.permutation(5)
sampler

df.take(sampler)

# 더미변수
df = pd.DataFrame({'key': ['b','b','a','c','a','b'],
                'data1': range(6)})

pd.get_dummies(df['key'])  #df에서 key라는 컬럼의 변수들을 dummy변수로 잡음.
dummies = pd.get_dummies(df['key'], prefix='key') # 접두어 붙여주기 가능
df_with_dum = df[['data1']].join(dummies) # join은 series가 아닌 DataFrame에 적용되는 함수. data[""]:Series, data[[""]]:DF
type(df[['data1']]);type(df['data1'])

df_with_dum

movies = pd.read_table('python_book/datasets/movielens/movies.dat', 
            sep="::", header=None, names = ['movie_id','title','genres'], engine='python')

movies[:10]
all_genres = []
type(movies)
for x in movies["genres"]: # = movies.genres
    all_genres.extend(x.split('|'))

genres = pd.unique(all_genres)
genres

zero_mat = np.zeros((len(movies), len(genres)))
dum = pd.DataFrame(zero_mat, columns = genres)
dum
movies.genres
gen = movies.genres[0]
gen.split('|')
dum.columns.get_indexer(gen.split('|'))

for i, gen in enumerate(movies.genres):
    indices = dum.columns.get_indexer(gen.split('|'))
    dum.iloc[i,indices]=1


movies_windic = movies.join(dum.add_prefix('Genre:'))
movies_windic.iloc[0]


import seaborn as sns
iris=sns.load_dataset('iris')
tips = sns.load_dataset('tips')

tips

tips.replace({'Dinner':'D', 'Lunch':'L'})

tips >> group_by(X.time) >> summarise(m = X.total_bill.mean().round(2))



list(range(0,10))
np.array(10)

a =[]

for i in range(0,5):
    a.append("{}".format(i))

url_base = "https://www.chicagomag.com"
url_sub = "/Chicago-Magazine/November-2017/Chicagos-Best-Steakhouses-2017/"
url = url_base + url_sub


from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.parse import urljoin
html = urlopen(url)
soup = BeautifulSoup(html , 'html.parser')
lists=soup.find('div',{'class':'steak-list'}).find_all('a')
len(list)

# 변수 속성 바꾸기
astype(float) #str,int


## 계층적 색인

# series
data = pd.Series(np.random.randn(9), index = [['a','a','a','b','b','c','c','d','d'],
                                                [1,2,3,1,3,1,2,2,3]])

data
data.index

data.unstack()

data['b']
data['b':'c']

data.loc[['b','d']]

# DataFrame
frame = pd.DataFrame(np.arange(12).reshape((4,3)),
                    index = [['a','a','b','b'],[1,2,1,2]],
                    columns= [['Ohio','Ohio','Colorado'],['Green','Red','Green']])

frame.index.names = ['key1', 'key2']
frame.columns.names = ['states','colors']
frame['Ohio']

# 계층의 순서 바꾸고 정렬하기: 계층의 순서를 바꿈.
frame.swaplevel('key1','key2')
frame.swaplevel(0,1).sort_index(level=1)

# 계층별 요약통계
frame.sum(level = 1, axis=1)

# DataFrame의 컬럼 사용하기
frame = pd.DataFrame({'a': range(7), 'b': range(7,0,-1),
                    'c': ['one','one','one','two','two','two','two'],
                    'd': [0,1,2,0,1,2,3]})

list(range(0,7,))
frame2 = frame.set_index(['c','d'], drop=False) # drop 여부에 따라 index로 설정한 columns 들이 남아있는지 없어지는지 설정
frame2.reset_index()

## 데이터 합치기(merge, join):merge, join의 경우 공통된 column이 있을 경우에 가능

df1 = pd.DataFrame({'key':['b','b','a','c','a','a','b'], 'data1':range(7)})
df2 = pd.DataFrame({'key': ['a','b','d'], 'data2': range(3)})

pd.merge(df1, df2, on = 'key') #merge on 

df3 = pd.DataFrame({'lkey':['b','b','a','c','a','a','b'], 'data1':range(7)})
df4 = pd.DataFrame({'rkey': ['a','b','d'], 'data2':range(3)})
pd.merge(df3, df4, left_on='lkey', right_on='rkey') # left_on: 조인키로 사용할 왼쪽 df의 컬럼
pd.merge(df1, df2, how='outer') 
# inner: 양 테이블 모두에 존재하는 키 조합 사용(공통). 
# left/right: 한쪽에 존재하는 모든 키 조합.
# ouput: 양쪽 테이블에 존재하는 모든 키 조합.

# 다대다 병합
df1 = pd.DataFrame({'key': ['b','b','a','c','a','b'], 'data1': range(6)})
df2 = pd.DataFrame({'key': ['a','b','a','b','d'], 'data2': range(5)})

pd.merge(df1, df2, on = 'key', how='left')

# 여러개의 키 병합하기
left = pd.DataFrame({'key1' : ['foo','foo','bar'], 'key2': ['one','two','one'], 'lval':[1,2,3]})
right = pd.DataFrame({'key1' : ['foo','foo','bar','bar'], 'key2': ['one','one','one','two'], 'rval':[4,5,6,7]})

merge1=pd.merge(left, right, on=['key1','key2'], how = 'outer')
pd.merge(left, right, on='key1', suffixes=('_left','_right')) # 이름이 동일한 column에 대해 suffix 붙여주기

merge1.fillna() # 결측치 단일값 채워넣기
merge1.interpolate(method='linear')

# 색인 병합하기
left1 = pd.DataFrame({'key': ['a','b','a','a','b','c'], 'value':range(6)})
right1 = pd.DataFrame({'group_val':[3.5,7]}, index=['a','b'])
left1
right1

pd.merge(left1, right1, left_on = 'key', right_index=True) # left에서 조인할 column 선택하고 right에서 index변수 선택


## 축 따라 이어붙이기(concatenation). 공통된 column 없어도 이어붙이기 가능

arr = np.arange(12).reshape((3,4))
arr
np. concatenate([arr,arr], axis=1) # axis=1, 열방향

s1 = pd.Series([0,1], index=['a','b'])
s2 = pd.Series([2,3,4], index=['c','d','e'])
s3 = pd.Series([5,6], index = ['f','g'])

#Series
pd.concat([s1,s2,s3])
pd.concat([s1,s2,s3], axis=1) # axis=1 새로운 열로 붙이기
s4=pd.concat([s1,s3])
pd.concat([s1,s4], axis=1, join='inner') # concat으로 이어붙이기

# DataFrame
df1= pd.DataFrame(np.arange(6).reshape(3,2), index=['a','b','c'], columns = ['one','two'])
df2= pd.DataFrame(5+np.arange(4).reshape(2,2), index=['a','c'], columns = ['three','four'])
df1
df2
pd.concat([df1,df2], axis=1, keys=['lev1','lev2']) # level로 columns의 계층 만들 수 있음.

df1= pd.DataFrame(np.arange(12).reshape(3,4), columns=['a','b','c','d'])
df2= pd.DataFrame(5+np.arange(6).reshape(2,3), columns=['b','d','a'])
df1
df2
pd.concat([df1,df2], ignore_index=True) # index가 없는 경우.


## pivoting


data = pd.read_csv('Python_book/examples/macrodata.csv')
data.info
data.head()

periods= pd.PeriodIndex(year = data.year, quarter = data.quarter, name= 'date')
columns = pd.Index(['realgdp','infl','unemp'], name='item')
columns
periods
data = data.reindex(columns=columns)
data.index = periods.to_timestamp('D','end')
ldata = data.stack().reset_index().rename(columns={0:'value'})
ldata
ldata.pivot('date','item','value')

df = pd.DataFrame({'key': ['foo','bar','baz'],
                'A':[1,2,3],
                'B':[4,5,6],
                'C':[7,8,9]})

melted = pd.melt(df,['key'])
melted
reshaped = melted.pivot('key','variable','value')
reshaped