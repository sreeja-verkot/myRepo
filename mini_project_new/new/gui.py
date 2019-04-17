from tkinter import *
from tkinter import messagebox
import backend_gui

firstclick = True

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
        Label(window, text =mood).grid(row=3)
window = Tk()
window.configure(background='white')
window.title("EMOJI PREDICTOR")
label = Label(window, text = "DeepMoji has learned to understand emotions. Type a sentence to see what our AI algorithm thinks.",fg ="green",font = ('Comic Sans MS',15),bg="white").grid(row=0)
#lab = Label(window, text="Text Message",bg="white",fg='red',font = ('Comic Sans MS',15)).grid(row=2,column=0)
e1 = Text(window,height=2, width=40)
e1.grid(row=2,column=0)
e1.insert(END, "type a sentence...")
e1.config(fg = 'grey')
e1.bind('<FocusIn>', on_entry_click)
Button(window, text='Submit', command=show_entry_fields,bg="green").grid(row=2,column=1, sticky=W)
window.mainloop()
