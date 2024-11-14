import requests
from bs4 import BeautifulSoup

url="https://tr.azrhymes.com/?"

t="tekerlemeler"

kelime=input("Kelime Giriniz:")
search_url=url+t+"="+kelime
content=requests.get(search_url).content

soup=BeautifulSoup(content,"html.parser")


def get_rhyme(soup):
	r=soup.find_all("span",{"class":"result"})
	
	return r

kafiyeler=get_rhyme(soup)

ilk_10=kafiyeler[:10]


dizi=[]
for i in ilk_10:
	dizi.append(i.text)

#print(dizi)

#dizideki , karakterleri silme 
for i in dizi:
	i=i.split(',')[0]
	print(i)