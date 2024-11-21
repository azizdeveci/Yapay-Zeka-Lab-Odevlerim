import requests
from bs4 import BeautifulSoup
import random

def resim_download():

	data=requests.get("https://tr.pinterest.com/ahmetyenal/galatasaray/").content
	data=BeautifulSoup(data,"html.parser")
	resimler=data.find_all("img")

	for i in resimler:
		print("Resmin açıklaması:",i["alt"])
		print("Resmin linki:",i["src"])
		acıklama=i["alt"]
		resim_linki=i["src"]

		# Resmin ismini acıklama.png olarak kaydediyoruz
		if acıklama:
			resim_data=requests.get(resim_linki).content
			with open(f"gs/{acıklama}.jpeg","wb") as file:
			    file.write(resim_data)

			
		#Açıklama kısmı boş ise random olarak sayı atıyoruz
		else:

			sayi=random.randint(0,99999)
			resim_data=requests.get(resim_linki).content
			with open(f"gs/{sayi}.png","wb") as file:
			    file.write(resim_data)

			
resim_download()

