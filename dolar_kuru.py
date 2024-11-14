from bs4 import BeautifulSoup
import requests
import time
import numpy as np
import matplotlib.pyplot as plt

class Dolar():
	def __init__(self)->None:
		self.dizi=[]
		self.veri_cek()
		

		dizi=np.array(self.dizi)
		plt.title("Dolar Kuru1")
		plt.plot(dizi,label="Dolar Kuru",color="red")
		plt.grid(True)
		plt.legend()
		plt.show()

	def veri_cek(self):
		for i in range(1,10):
			data=requests.get("https://kur.doviz.com/serbest-piyasa/amerikan-dolari").content
			data=BeautifulSoup(data,"html.parser")
			dolar_kur=data.find("div",{"class":"flex justify-between mt-8"})
			dolar_kur=dolar_kur.find("div",{"data-socket-key":"USD"}).text
			self.dizi.append(dolar_kur)

			print(dolar_kur)
			time.sleep(0)
 

t=Dolar()
