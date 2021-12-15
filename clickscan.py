from tkinter import *
from tkinter import ttk
from datetime import datetime
import tkinter

clickCount = 0
timeStarted = 0
clickArr = []

def resetClicks():
    global clickCount, clickArr, timeStarted
    clickCount = 0
    clickArr = []

def click():
    global clickCount, clickArr, timeStarted
    if clickCount >= 50:
        save()
        resetClicks()

    if clickCount == 0:
        timeStarted = datetime.now().timestamp()
    
    clickCount+=1
    clickArr.append(
        round(
            datetime.now().timestamp() - timeStarted,
            3
        )
    )

def save():
    global clickCount, clickArr, timeStarted
    if clickCount > 0:
        selectedType = list.get(list.curselection())
        file = open(f'clicks/{selectedType}/{clickCount}_{timeStarted}.txt', 'w')
        file.write(str(clickArr))
        print(f'Saved {clickCount} clicks to {selectedType}')

root = Tk()
root.attributes('-topmost', True)
frm = ttk.Frame(root, padding=10)
frm.grid()

ttk.Button(frm, text="CLICK HERE", command=click).grid(
    column=0, row=0, ipady=50,ipadx=50
)

list = Listbox(frm, width=28, height=8)
list.grid(column=0, row=1, rowspan=4)

x = ['complex', 'human', 'simple']
for type in x:
    list.insert(END, type)

root.mainloop()