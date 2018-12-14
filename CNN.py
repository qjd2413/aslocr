import random
import string

string = 'YOU ARE A CUCK'
index = 0

def predict(frame):
    global string, index
    if index >= len(string):
        return None
    char = string[index]
    index += 1
    return char
