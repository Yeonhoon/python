url = []
x=[]
for item in lists:
    url.append(urljoin(url_base, item['href']))

x = []
for i in range(len(lists)):
    x.append(lists[i].get_text().split("."))

df= pd.DataFrame(x, columns = ['Number','Store']).set_index('Number')
df['url']= url
df


data = {"Store":x, "URL":url}
df = pd.DataFrame(data)
df['url'][0]
type(url)
html = urlopen(url)
url
html = urlopen(url)

soup_steak=BeautifulSoup(html, 'html.parser')
soup_steak.find_all('section',{'class':'related-content pull-right'})
a.find_all('aside',{'class':'related-content pull-right'})
a.find_all('section')
len(address)



soup_steak = BeautifulSoup(urlopen(df['url'][0]), 'lxml')
txt_tmp=soup_steak.find('section','related-content pull-right').find('ul').get_text()
txt_tmp.split('\n')[1:-3]
txt_tmp.split('\n')[-3]


a=BeautifulSoup(urlopen(df['url'][1]), 'lxml')
st=soup_steak.find_all('section')
bool(a.find_all('aside','related-content pull-right'))
bool(a.find_all('section','related-content pull-right'))

from tqdm import tqdm_notebook
soup_steak('aside')

address=[]
phone = []



for i in df.index:
    html = urlopen(df['url'][i])
    soup_steak = BeautifulSoup(html, 'lxml')
    if bool(soup_steak.find('section',{'class':'related-content pull-right'})) == True :
        address_tmp= soup_steak.find('section',"related-content pull-right").find('ul').get_text()
    else :
        address_tmp = soup_steak.find('aside',"related-content pull-right").find('ul').get_text()
    address.append(''.join(address_tmp.split('\n')[1:-3]))
    phone.append(''.join(address_tmp.split('\n')[-3]))

len(address), len(phone)
address
df['url'][1]    
df['address'] = address
df['phone_number'] = phone
df.head()
df = df.loc[:, ["Store","address",'phone_number','url']]
df

