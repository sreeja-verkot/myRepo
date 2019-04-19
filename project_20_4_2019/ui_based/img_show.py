from tkinter import *
from PIL import Image, ImageTk
# pip install pillow
def show_img(image):
    #root = Tk()
    #root.wm_title("Tkinter window")
    #root.geometry("1000x1000")
    load = Image.open(image)
    render = ImageTk.PhotoImage(load)
    img = Label(image=render)
    img.image = render
    img.place(x=100, y=100)
    #root.mainloop()

