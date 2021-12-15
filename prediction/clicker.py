# Simple clicker with no randomization at all because I'm not
# downloading anything coming from sourceforge

import pyautogui
from time import sleep

width, height = pyautogui.size()

while(True):
    pyautogui.click()
    sleep(0.03)

