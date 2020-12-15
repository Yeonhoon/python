# requests: 텍스트든 뭐든 가져오기
import requests

#beautifulSoup: 필요한 텍스트 빼오기
from bs4 import BeautifulSoup
LIMIT = 50
URL = (f"https://kr.indeed.com/%EC%B7%A8%EC%97%85?as_and=python&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=all&st=&salary=&radius=25&l=&fromage=any&limit={LIMIT}&sort=&psf=advsrch&from=advancedsearch")

def get_last_page():
    result = requests.get(URL)
    # 필요한걸 뽑아오기 위한 soup 만들어주기:html 전부 가져오기
    soup = BeautifulSoup(result.text, "html.parser")
    # 페이지: html에서 class:pagination인 요소들을 찾기.
    pagination = soup.find("div",{"class":"pagination"})
    # pagination에서 'a' 모두 찾기
    links = pagination.find_all('a')
    pages= []
    for link in links[:-1]:
        pages.append(int(link.string)) # string 요소(텍스트)만 뽑기
    last_page = pages[-1]
    return last_page
    

def extract_job(html):
    #anchor(a) 에서 "title" 추출하기
    title = html.find("h2",{"class": "title"}).find("a")["title"]
    company = html.find("span", {"class":"company"}) 
    company_anchor = company.find("a")
    # 링크가 있는 회사가 있고 없는 회사가 있어서 none이 뜨기 때문에 if/else로 구분하여 뽑아내기.
    if company_anchor != None:
        company = str((company_anchor.string))
    else:
        company = str(company.string)
    #strip(): 괄호 안에 들어가는 단어로 시작하는 요소 벗기기(없애기)
    company = company.strip()
    # location: job들의 위치를 html에서 찾아오기
    location = html.find("div", {"class":"recJobLoc"})["data-rc-loc"]
    # job_id: 누르면 그 페이지로 연결되는 직무의 id 찾기
    job_id = html["data-jk"]
    return {'title': title, 'company': company, 'location': location,
    "link":f"https://kr.indeed.com/%EC%B7%A8%EC%97%85?as_and=python&as_phr&as_any&as_not&as_ttl&as_cmp&jt=all&st&salary&radius=25&fromage=any&limit=50&sort&psf=advsrch&from=advancedsearch%22&vjk={job_id}"}


## 하나의 function이 길게 늘어지는 것보다 여러 개의 function으로 쪼개기


def extract_jobs(last_page):
    # 페이지별 직업 리스트 개수 표기
    jobs = []
    for page in range(last_page):
        print(f"Scrapping page {page}")
        result = requests.get(f"{URL}&start={page*LIMIT}")
        soup = BeautifulSoup(result.text, "html.parser")
        results = soup.find_all("div", {"class": "jobsearch-SerpJobCard unifiedRow row result"})
        for result in results:
            job = extract_job(result)
            jobs.append(job)
    return jobs


def get_jobs():
    last_page = get_last_page()
    jobs = extract_jobs(last_page)
    return jobs




indeed_jobs = get_jobs()
print(indeed_jobs)




