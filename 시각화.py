# 좀더 편리한 시각화 도구
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns  # seaborn을 불러올때는 항상 matplotlib도 같이 import 해야함.


## 한글깨짐 해결하기
import platform

from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus']=False

if platform.system() == "Windows":
    path = "C:\Windows\Fonts/NanumGothic.ttf"
    font_name = font_manager.FontProperties(fname = path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown System...')
%matplotlib inline


x= np.linspace(0,14,100) #0~14까지 100개 무작위로
y1= np.sin(x)
y2=2*np.sin(x+0.5)
y3=3*np.sin(x+1.0)
y4=4*np.sin(x+1.5)

sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.plot(x,y1,x,y2,x,y3,x,y4)
plt.show()

##seaborn
flights=sns.load_dataset('flights')
tips=sns.load_dataset('tips')
plt.figure(figsize=(8,6))

#boxplot
sns.boxplot(x='day', y='total_bill', data=tips)
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='day', y='total_bill', data=tips, hue='smoker', palette='Set3') #hue: 그룹 나누기.
plt.show()

sns.set_style('darkgrid')

#lmplot
sns.lmplot(x='total_bill', y='tip', data=tips, size=7, hue='smoker', palette='Set1')
plt.show()

#pairpltot
iris=sns.load_dataset('iris')
sns.set(style='ticks')
sns.pairplot(iris,palette='husl', hue='species',diag_kind="hist", corner=True)  #corner, diag_kind
plt.show()

#heatmap
sns.heatmap(crime_norm_sort[target_col], annot=True, fmt='f', linewidths=0.5, cmap='magma_r')
cmap = 'viridis','plasma','inferno','magma','cividis' # 등등.. 뒤에 _r 붙이면 색상 위아래 뒤집힘.

# 시계열 그래프
sns.lineplot(data=flights)


#folium
import folium
map_osm = folium.Map(location = [45.5236,-122.6750], zoom_start=13,
                    tiles = 'Stamen Toner')
folium.Marker([45.5244,-122.6699], popup ="The WaterFront").add_to(map_osm) # 마커 표시하기
folium.CircleMarker([45.5215,-122.6261], radius=50,
                       popup = "Laurenlhurst Park", color = "#3186cc",
                       fill_color="#3186cc").add_to(map_osm) # 원 표시하기

#tiles 종류:
#”OpenStreetMap”
# ”Stamen Terrain”, “Stamen Toner”, “Stamen Watercolor”
# ”CartoDB positron”, “CartoDB dark_matter”
# ”Mapbox Bright”, “Mapbox Control Room” (Limited zoom)