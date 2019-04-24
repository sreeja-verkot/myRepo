from tkinter import *
from tkinter import messagebox
import backend_gui
#import img_show
from PIL import Image, ImageTk

firstclick = True
emoji_dict = {"joy":"joy.jpg", "fear":"fear.jpg", "anger":"anger.jpg", "sadness":"sad.jpg", "disgust":"disguist.jpg", "shame":"shame.jpg", "guilt":"guilt.jpg"," ":"blank.jpg"}
def imgpress():
    global img
    img.destroy()
    #Label1 = Label(window, text="Image has been cleared",fg="#0094FF",font=('Arial',15)).grid()
    return;
def show_img(image):
    global img
    #root = Tk()
    #root.wm_title("Tkinter window")
    #root.geometry("1000x1000")
    load = Image.open(image)
    load = load.resize((150, 150), Image.ANTIALIAS) 
    render = ImageTk.PhotoImage(load)
    img = Button(window, image=render,command=imgpress)
    img.image = render
    img.place(x=500, y=570)
    #root.mainloop()

def on_entry_click(event):
    """function that gets called whenever entry1 is clicked"""        
    global firstclick

    if firstclick: # if this is the first time they clicked it
        firstclick = False
        e1.delete('1.0', END) # delete all the text in the entry
def show_entry_fields():
    if e1.get("1.0",'end-1c')=="type a sentence..." or e1.get("1.0",'end-1c')=="":
        messagebox.showerror("error", "Please enter a sentence")
        return
    else:
        mood=backend_gui.test(e1.get("1.0",'end-1c'))
        Label(window, text =mood[0],bg="white").grid(row=5,sticky=W+E+N+S)
        show_img(emoji_dict.get(mood[0]))
window = Tk()
photo = PhotoImage(file = "bg.gif")
w = Label(window, image=photo).grid()
window.configure(background='white')
window.title("EMOJI PREDICTOR")
label = Label(window, text = "DeepMoji has learned to understand emotions. Type a sentence to see what our AI algorithm thinks.",fg ="green",font = ('Comic Sans MS',15),bg="white").grid(row=2)
#lab = Label(window, text="Text Message",bg="white",fg='red',font = ('Comic Sans MS',15)).grid(row=2,column=0)
e1 = Text(window,height=2, width=40)
e1.grid(row=4,column=0)
e1.insert(END, "type a sentence...")
e1.config(fg = 'grey')
e1.bind('<FocusIn>', on_entry_click)

Button(window, text='Submit', command=show_entry_fields,bg="green").grid(row=4,column=1, sticky=W)
window.mainloop()
