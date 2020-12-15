## numpy: 파이썬 리스트보다 더 빠른 연산속도, 더 적은 메모리량 차지.
import numpy as np

data = np.random.randn(2,3)

# ndarray: 같은 종류의 데이터를 담을 수 있는 포괄적인 다차원 배열.
# shape: 차원의 크기를 알려줌
# dtype: 배열에 저장된 자료형을 알려줌

data1 = [6,7.5,8,0,1]
arr1 = np.array(data1) 


# 다차원 배열
data2 = [[1,2,3,4],[5,6,7,8]]
arr2 = np.array(data2)
arr2[1]
arr2.ndim

arr1.dtype #float64: 부동소수점
arr2.dtype #int32: 정수

# zeros: 0의 배열 생성
np.zeros(10)
np.zeros((3,6))
np.zeros((2,3,2))
np.arange(15)

np.ones(10)

arr = np.array([1,2,3,4,5])
arr.dtype
float_arr = arr.astype(np.float64)
float_arr.dtype

arr = np.array([[1,2,3],[4,5,6]])

# 색인과 슬라이싱

arr = np.arange(10)
arr[5:8] = 12
arr

# arr에서 슬라이스를 한 뒤 arr에서 slice한 객체(arr_slice)의 숫자를 변경하면 arr에도 반영이 되어있음.
arr_slice = arr[5:8]
arr_slice
arr_slice[1]=12354
arr_slice[:] = 64
arr

arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d[0][2]
arr2d[0,2] # 동일한 표현방식.

arr2d[:2,1:]
arr2d[:,:1]

names = np.array(['Bob',"John",'Joe','Will','Joe','Bob','Will'])
data = np.random.randn(7,4)
names
names == 'Bob'
data[~(names =='Bob')] # ~: 조건 반대로 쓰기.
mask = (names =="Bob") | (names == "Will")

data[mask]
data[data<0]=0
data[names !='Joe']=7
data

## 팬시(fancy) 색인
#np.empty(()): 빈 리스트 만들기
np.empty((8,8))
arr= np.empty((8,4))
for i in range(8):
    arr[i] = i
    arr
arr
arr[[1,3,2,5]]

# 축 바꾸기(x,y 뒤집기)
arr.T

# 유니버셜 함수 배열의 각 원소를 빠르게 처리하는 함수
# 단항 유니버셜 함수
arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)

# 다항 유니버셜 함수
x=np.random.randn(8)
y= np.random.randn(8)
y
np.maximum(x,y) #원소별로 가장 큰 값 계산

#np.arange(x,y,z): x부터 y까지 z의 단위로
points = np.arange(-5,5,0.01) #-5부터 4.99까지 0.01씩 증가하는 값들의 배열

xs, ys = np.meshgrid(points, points) #meshgrid: 두개의 1차원 배열을 받아서 가능한 모든 x,y 짝을 만들 수 있는 2차원 배열 두개 반환
z=np.sqrt(xs **2 + ys **2) # **: 승(square)
z

import matplotlib.pyplot as plot
plt.imshow(z, cmap= plot.cm.gray); plt.colorbar()
plt.title("Image plot of $\sqrt{X^2 + y^2}$ for a grid of values")
plot.show()


# 배열 연산으로 조건절 표시하기.

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y)
        for x,y,c in zip(xarr, yarr, cond)]
result

arr = np.random.randn(4,4)
arr
arr>0

#np.where : ifelse
np.where(arr >0,2,-2) # 양수는 2로 바꾸고 음수는 -2로 바꾸기.
np.where(arr>0, 2, arr) # 음수는 그대로 arr

# np에서 통계 산출

arr= np.random.randn(5,4)
arr
np.sum(arr)
np.mean(arr) # = arr.mean()
np.std(arr)

arr.mean(axis=0) #0: 행(rows)
arr.mean(axis=1) #1: 열(columns)

arr = np.array([0,1,2,3,4,5,6,7])
arr.cumsum()

arr = np.array([[0,1,2],[3,4,5],[6,7,8]])
arr.cumsum(axis=1)

# boolean을 위한 방법
arr = np.random.rand(100)
arr
(arr>5).sum()
bools = np.array([False, False, True, False])
bools.any() # array 중에 하나라도 True가 있는가?
bools.all() # 모두 true인가?

# 정렬

arr = np.random.rand(6)
arr.sort()
arr

arr = np.random.rand(5,3)
arr.sort(0)
arr

# 집합관련 함수
names = np.array(['bob','joe','will','bob','jane','joe','will','dan'])
np.unique(names)
np.in1d(names,'bob') # names에 'bob'이 있는지 t/f로 알려줌.

# 배열 데이터의 파일 입출력

arr = np.arange(10)
brr = np.arange(20)
np.save('some_array',arr)
np.load('some_array.npy')

#savez: 여러 배열을 압축하여 저장.
np.savez('array_archive.npz', a=arr, b= brr)
arch = np.load('array_archive.npz')
arch['a']

# 선형대수

x= np.array([[1., 2., 3.,],[4., 5., 6.,]])
y = np.array([[6.,23.], [-1,7],[8,9]])

x
y
np.dot(x,y) # 행렬의 곱셈

x @ np.ones(3) #np.ones: 모든 값이 1인 배열, @: 행렬간 곱.

np.zeros(3)

from numpy.linalg import inv, qr
X= np.random.randn(5,5)
mat= X.T.dot(X)  #.T: 전치행렬
inv(mat) #inv: 역행렬
mat

# 난수 생성
samples = np.random.normal(size=(4,4)) # normal: 표준 정규분포
samples

np.random.seed(1234)

import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0,1) else -1 #randint: 주어진 최소/최대범위 안에서 임의의 난수 추출.
    position +=step
    walk.append(position)

plt.plot(walk[:100])
plt.show()

nsteps= 1000
draws = np.random.randint(0,2, size= nsteps)
steps = np.where(draws> 0, 1, -1) # 조건에 따라 양수 음수 나누기.
walk = steps.cumsum()
walk.min()
walk.max()

nwalks = 5000
nsteps = 1000

draws = np.random.randint(0,2, size = (nwalks, nsteps))
draws
steps = np.where(draws>0,1,-1)
steps
walks = steps.cumsum(1)
walks

hits30 = (np.abs(walks)>=30).any(1)
hits30
hits30.sum()