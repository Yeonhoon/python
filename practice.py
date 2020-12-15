# print(-5)

# animal = "cat"
# name = "JYH"
# age = 21
# hobby = "Health"
# is_adult = age > 19
# '''
# print("내 이름은", name, "이고 나이는", age, "이고 취미는", hobby, "이다")
# ''' # ctrl + / : 주석처리

# # 변수명: Station

# # 변수값: "사당", "신도림", "인천공항" 순서대로 입력

# station = ["사당","신도림","독산"]

# for i in station:
#     print(i, "행 열차가 들어오고 있습니다.")

# from random import *
# print(random())

# print(randrange(1,46))

# date = randint(4, 28)



# print("오프라인 스터디 모임 일자는 매월", str(date),"로 선정되었음")


# # 단축키: ctrl + D: 단어 선택하기(반복하여 누르면 같은 단어 한번에 여러개 변경 가능)
# # alt + up/down: 코드 위아래로 이동
# # alt + shift + up/down: 위아래로 복사
# # alt + shift + i : 여러 줄 동시에 편집 가능
# # alt + shift + drag    



# python = "python is amazing"
# print(len(python))
# print(python.replace("python","R"))
# indx = python.index("n", indx +1)
# print(indx)
# print(python.find("java")) # 원하는 값이 없으면 -1 표기함.
# print(python.index("java")) #원하는 값이 없으면 오류남

# # 문자열 합치는 방법: "+", ","
# # 방법 1
# print("나는 %d살입니다" % 20) 
# print("나는 %s살입니다" % "파이썬")
# print("나는 %c살입니다" % "A")
# print("나는 %s색과 %s색을 좋아합니다." %("파란","분홍"))

# #방법 2
# print("나는 {}살입니다.".format(20))
# print("나는 {}색과 {}색을 좋아하는 {}입니다.".format("레드","블랙","JYH"))

# # 방법 3
# print("나는 {1}색과 {0}색을 좋아하는 {2}입니다.".format("레드","블랙","JYH"))
# print("나는 {age}살이며 {color}색을 좋아함" .format(color = "red", age = "33"))

# # 방법 4
# age = 20
# color = "red"
# print(f"나는 {age}살이며 {color}색을 좋아함")

# # 줄바꿈: \n
# print("안녕하세요 \n 장연훈입니다")
# print("안녕하세요 저는 '장연훈'입니다")
# print("안녕하세요 저는 \"장연훈\"입니다") #\" ~ \"

# # \b: 백스페이스
# print("redd\bapple")
# # \t: tab
# print("red\tapple")

# site = "http://google.com"
# site2 = site.replace("http://","")
# site2
# site3 = site2[: site2.index(".")]
# pw = site3[:3] + str(len(site3)) + str(site3.count("e")) + str("!")
# print(f"{site} 의 비밀번호는 {pw} 입니다")

# #################### list: [] ###########################
# subway = ["장","임","박","최","홍"]
# print(subway.index("홍"))

# #append: 하나씩 추가할 때.
# subway.append("김")

# #insert
# subway.insert(2, "박")
# print(subway)

# #pop: 뒤에서 하나 빼기
# print(subway[:3])
# print(subway.pop())
# print(subway)

# #count: 리스트 내부 수 세기
# print(subway.count("박"))

# # sort: 정렬
# num_list = [5,4,3,2,1]
# print(num_list.sort())

# # reverse: 순서 뒤집기

# #clear: 지우기
# print(num_list.clear())

# #extend: 리스트 확장
# print(num_list)
# max_list=["sdfsd","Sdfsd"]
# num_list.extend(max_list)
# print(num_list)
# num_list.append("hi") 

# ######################## dictionary: {} #########################

# cabinet = {"A-1": "장연훈", "Z-1": "홍미령"}
# print(cabinet["A-1"])
# print(cabinet.get(10,"사용 가능"))
# print(3 in cabinet) #boolean

# # 추가
# cabinet["B-1"] = "훈련장"
# print(cabinet["B-1"])

# del cabinet["B-1"]
# print("B-1" in cabinet)

# #key 들만 출력
# print(cabinet.keys())

# #values
# print(cabinet.values())

# #keys & values
# print(cabinet.items())

# # 비우기
# cabinet.clear()
# print(cabinet)

# ####################### tuple: () 리스트처럼 변경(추가,제거)은 안되지만, 리스트보다 빠름 ############################
# menu = ("떡볶이","순대")
# print(menu[0])

# name = "장"
# age = 27
# hobby = "training"
# print(name, age, hobby)
# (name, age, hobby) = ("장",289,"training") 

# ####################### set(집합): {} 중복 안됨, 순서 없음.
# my_set = {1,1,2,2,3,3,4}
# print(my_set)

# java = {"a","b","c"}
# python = set(["a","c"])

# # 교집합
# print(java & python)
# print(java.intersection(python))

# # 합집합
# print(java | python)
# print(java.union(python))

# # 차집합
# print(java - python)
# print(java.difference(python))

# # 추가: add
# java.add("s")
# print(java)

# ######### 자료 구조의 변경 ##############
# menu = {"커피","우유","스무디"}
# print(menu, type(menu))

# menu = list(menu)
# print(menu, type(menu))

# menu = tuple(menu)
# print(menu, type(menu))

# from random import *
# from builtins import list
# list = [1,2,3,4,5]
# del list
# shuffle(list)
# list
# users = range(1, 20)
# print(type(users))
# users = list(users)
# print(type(users))
# shuffle(users)
# print(users)
# winner = sample(users, 4)
# print(f"치킨 당첨자: {winner[0]}")
# print(f"커피 당첨자: {winner[1:4]}")


################################### if ###################
# weather = input("오늘 날씨는 어떠냐? ")
# if weather == "비":
#     print("우산 챙겨라")
# elif weather == "미세먼지":
#     print("마스크 챙겨라")
# else:
#     print("맨몸으로 나가라")


# temp = int(input("오늘의 기온은 어떠냐? "))
# if temp >= 30:
#     print("더워 뒤짐")
# elif 0 <= temp < 10:
#     print("롱패딩 준비해라")
# elif 10<= temp <30:
#     print("괜춘")
# else:
#     print("얼어 디짐")


##################### for ##################################################
waitings = list(range(1,50))
for waiting_no in waitings:
    
    print(f"{waiting_no} 대기번호 준비완료")

starbucks = ["iron man","Thor","Captain", "Hulk", "Vision", "Scarlet Witch"]

for i in starbucks:
    print(f"{i}님 주문하신 아메리카노 나왔습니다.")


#################### while ########################## 조건을 만족할 때 까지 조건문을 계속 반복.
# customer = "Thor"
# index = 5
# while index >=1:
#     print(f"{customer}님 커피 준비됐습니다. 부르는거 {index}번 남았다.")
#     index -=1 # +=1 --> 무한 루트, ctrl + c 누르면 스탑.
#     if index ==0:
#         print("커피 먹지마 그냥")


# continue & break

absent = [2,5]
no_book = [7]
for student in range(1,11):
    if student in absent:
        continue
    elif student in no_book:
        print(f"{student}는 제정신이냐? 교무실로 따라와라")
        break
    print(f"{student}번 학생 책 읽어봐라")


student = [1,2,3,4,5]
print(student)
students = [i+ 100 for i in student]
print(students)

Marvel_hero = ["Ironman","Captain America","Thor","Hulk","Spiderman", "Vison"]
marvel = [len(i) for i in Marvel_hero]
print(marvel)

########## 퀴즈
from random import *
drive_time = list(range(5,50))
print(shuffle(drive_time))

customers = list(range(1,50))


customer=0
for i in range(1,51):
    time = randrange(5,51) ## randrange:: 랜덤 숫자 범위 선정
    if 5<= time <=15:
        print(f"[O] {i}번째 손님(소요시간: {time}분)")
        customer += 1 
    else:
        print(f"[] {i}번째 손님(소요시간: {time}분)")
    if i == 50:
        print("총 탑승 승객:",f"{customer} 분")


## [::숫자]: 리스트에서 해당 숫자만큼 띄워서 뽑기


# enumerate(): 순차 자료형에서 현재 아이템의 색인을 함께 처리

# sorted(): 

# zip: 두개의 리스트를 순차적으로 교차하여 묶어줌.
# enumerate와 종종 같이 사용
# e.g.
seq = [1,2,3]
seq2 = ["foo","bar","zoo"]
for i , (a,b) in enumerate(zip(seq,seq2)):
    print(f"{i}: {a}와 {b}")

# reversed
a=list(range(10))
list(reversed(a))
a[::-1]

words = ['apple','bat','bar','atom','banana','book','asshole']
by_letter = {}
words
for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter]=[word]
    else:
        by_letter[letter].append(word)

#{}(set): 집합

a = {1,2,3,4,5}
b = {1,3,5,7,9}
# .union:(|): 합집합
a.union(b)

#.intersection(&): 교집합

#difference: 차집합

strings = ['a','as','ass','bat','batman','cat','dove','python','R']
[x.upper() for x in strings if len(x)>2]

unique_length = {len(x) for x in strings}

unique_length

all_data = [['John','Emily','Michael','Mary','Steve'], ['Maria','Juan','Javier','Natalia','Pilar']]

result = [name for names in all_data for name in names if name.count('e')>=2]

# 리스트의 리스트 생성
some_tuples = [{1,2,3},{4,5,6},{7,8,9}]
[[x for x in tup] for tup in some_tuples]

# 함수에서 여러 개의 함수 반환하기
def f():
    a= 5
    b= 6
    c= 7
    return {"a": a,"b": b,"c": c} # 사전의 형태로 결과 반환.

# 필요없는 문장부호 제거하거나 대소문자 맞추기
import re
states = ["Alabama","Georgia","georgia","florida","SOUthcarolina##","West Virgina?"]
def clean_strings(strings):
    result =[]
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]','', value)
        value = value.title()
        result.append(value)
    return result

clean_strings(states)


def remove_punctuation(value):
    return re.sub('[!#?]','',value)

clean_ops = [str.strip, remove_punctuation, str.title]
clean_ops

def clean_strings(strings, ops):
    result: []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result

clean_strings(states, clean_ops)


for x in map(remove_punctuation, states):
    print(x)


# 익명함수: 람다(lambda)
# 이름을 지어주지 않고 함수 기능 나타내기. 데이터 분석에서 편리

# 기존의 함수방식
def short_function(x):
    return x*2

# 람다방식
equiv_anon = lambda x:x*2

strings
strings.sort(key = lambda x: len(list(x)))
sorted(strings, key= lambda x: len(list(x)))

# 커링: 일부 인자만 취하는 함수 만들기

def add_num(x,y):
    return x+y
add_five = lambda y: add_num(5,y)

# partial 함수로 커링
from functools import partial
add_five = partial(add_num, 5)

# 제너레이터
some_dict = {'a':1, 'b':2, 'c':3}
for key in some_dict: 
    print(key)

