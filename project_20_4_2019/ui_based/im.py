from tkinter import *
from PIL import Image, ImageTk

def imgpress():
    global img
    img.destroy()
    Label1 = Label(root, text="Image has been clicked",fg="#0094FF",font=('Arial',20)).pack()
    return;




# pip install pillow
root = Tk()
root.wm_title("Tkinter window")
root.geometry("1000x1000")
load = Image.open("bg.jpg")
load = load.resize((150, 150), Image.ANTIALIAS) 
render = ImageTk.PhotoImage(load)
img = Button(root, image=render,command=imgpress)
img.image = render
img.place(x=0,y=0)
root.mainloop()

