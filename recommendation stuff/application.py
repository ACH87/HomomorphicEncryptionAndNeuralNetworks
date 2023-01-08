from tkinter import *
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import recommendation_test
import Math_Functions as mf
import requests
import zipfile
from PIL import ImageTk, Image
import io
import math
import os

window = Tk()
window.title('Reverse Search')

window.geometry('808x710')
window.configure()

# entry - an image file path
entry = Entry(window, width=120)
entry.grid(column=0, row=0)
entry.focus()

# key = requests.get('http://192.168.0.17:8888/public_key').json()
# features = recommendation_test.findFeatures(r'C:\Users\saqla\Documents\Uni\Fourth Year\FYP\recommendation algorithm\dataset\images\1163.jpg')
# req = {'individual_features': mf.encrypt(features, key)}
# sum_sqrt = 0
# for x in features[0]:
#     sum_sqrt += int(x)
#
# req['sum_sqrt'] = sum_sqrt
# req['number_of_images'] = 5
# images = requests.post('http://192.168.0.17:8888/findSimilar', json=req).content
# print(images)
# z = zipfile.ZipFile(io.BytesIO(images))
# data = z.read(z.namelist()[0])
# dataEnc = data # Encode the raw data to be used by Image.open()
# img = Image.open(io.BytesIO(dataEnc))
p = None
panels = []
for x in range(5):
    panel = Label(window, image=p, anchor='center')
    panel.grid(column=0, row=x+3)
    panel.image = p
    panels.append(panel)
# panel_1 = Label(window, image= p)
# panel_1.grid(column=0, row=2)
# panel_1.image = p
# panel_2 = Label(window, image= p)
# panel_2.grid(column=0, row=2)
# panel_2.image = p
# panel_3 = Label(window, image= p)
# panel_3.grid(column=0, row=2)
# panel_3.image = p
# panel_4 = Label(window, image= p)
# panel_4.grid(column=0, row=2)
# panel_4.image = p
# panel_5 = Label(window, image= p)
# panel_5.grid(column=0, row=2)
# panel_5.image = p

def clicked(encrypted=True):
    path = entry.get()
    key = requests.get('http://192.168.0.17:8888/public_key').json()
    features = recommendation_test.findFeatures(path)
    sum_sqrt = 0
    for x in features[0]:
        sum_sqrt += (x**2)

    if encrypted:
        req = {'individual_features': mf.encrypt((features*100).tolist(), key),
               'sum_sqrt': mf.encrypt([[int(math.sqrt(sum_sqrt*100))]], key)[0][0], 'number_of_images': 5}
    else:
        req = {'individual_features': (features*100).tolist(), 'sum_sqrt': int(math.sqrt(sum_sqrt)*100), 'number_of_images': 5}

    req['encrypted'] = encrypted

    print('req', req)

    images = requests.post('http://192.168.0.17:8888/findSimilar', json=req).content
    print(images)
    for x in range(5):
        z = zipfile.ZipFile(io.BytesIO(images))
        data = z.read(z.namelist()[x])
        dataEnc = data # Encode the raw data to be used by Image.open()
        img = Image.open(io.BytesIO(dataEnc))
        p = ImageTk.PhotoImage(img)
        panels[x].configure(image=p)
        panels[x].image = p
    window.update()

def click_true():
    clicked()

def click_false():
    clicked(False)

convert_button = Button(window, text='Search', command=click_true, height=1, width=50)
convert_button.grid(column=0, row=2)

search_non_enc = Button(window, text='Search Non', command=click_false, height=1, width=50)
search_non_enc.grid(column=0, row=1)

window.mainloop()