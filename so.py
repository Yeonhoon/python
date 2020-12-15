# requests: 텍스트든 뭐든 가져오기
import requests

#beautifulSoup: 필요한 텍스트 빼오기
from bs4 import BeautifulSoup
LIMIT = 50
URL = "https://stackoverflow.com/jobs?q=python&pg=2"


def get_last_page():
    result = requests.get(URL)
    soup = BeautifulSoup(result.text, "html.parser")
    pages = soup.find("div",{"class":"s-pagination"}).find_all("a", {"class":"s-pagination--item"})
    print(pages)
    print(last_pages)


def get jobs():
    last_page = get_last_page()
    return []
    