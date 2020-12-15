def open_accout():
    print("새로운 계좌 생성됨")

def deposit(balance, money):
    print(f"입금 완료. 잔액은", f"{balance+money}원 입니다.")
    return balance + money

def withdraw(balance, money):
    if balance >= money:
        print("출금이 완료되었습니다. 잔액은",f"{balance - money}원 입니다.")
        return balance - money
    else:
        print("출금에 실패했습니다. 잔액은", f"{balance}원 입니다.")


balance = 0 
balance = deposit(balance, 10000)
print(balance)
withdraw(balance, 5000)

def profile(name, age, main_lang):
    print("이름: {0}, 나이: {1}, main_lang:{2}" .format(name, age, main_lang))

profile("장연훈", 26, "R")

# 가변 인자

def profiles(name, age, *languages):
    print("이름: {0}\t나이: {1}\t". format(name, age), end = " ")
    for i in languages:
        print(i, end = " ")
    print()

profiles("a", 22, "R", "C", "C++", "C#", "Django", "Python")

# 지역변수: 함수를 호출했을 때(def 내에서) 사용되는 변수

def checkpoints(soldiers):
    gun =20 
    gun = gun -soldiers
    print("[총기함] 남은 총: {}".format(gun))

checkpoints(2)


# 퀴즈

def std_weight(height, gender):
    if gender == "남자":
        weight = round(height * height * 22*0.0001, ndigits= 2)
        print("키 {0}cm인 {1}의 표준 체중은 {2}입니다.".format(height, gender, weight))
    else:
        weight = round(height * height * 21*0.0001, ndigits= 2)
        print("키 {0}cm인 {1}의 표준 체중은 {2}입니다.".format(height, gender, weight))

std_weight(175, "남자")


