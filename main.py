# This is a sample Python script.
import os
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
from load import load_mnist, extract_mnist



#Append custom module directories to sys.path
# sys.path.append(os.path.join(os.getcwd(), "load"))
#                 #,os.path.join(os.getcwd(), "data"),os.path.join(os.getcwd(), "config"), os.path.join(os.getcwd(), "misc")])
print(sys.path)



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

load_mnist.load_mnist(os.getcwd())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
