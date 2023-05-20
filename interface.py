import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
from keras.utils import img_to_array, load_img
import numpy as np
from tensorflow import keras

model = keras.models.load_model('VEHICLE.h5')
labels_vehicle = ['bicycle', 'boat', 'car', 'motorbike', 'airplane', 'train', 'truck']

classified_images_bicycle = []
classified_images_boat = []
classified_images_car = []
classified_images_motorbike = []
classified_images_airplane = []
classified_images_train = []
classified_images_truck = []


def open_image():
    global file_path 
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg")])
    if file_path:
        current_image = Image.open(file_path)
        current_image = current_image.resize((400, 300))
        image_tk = ImageTk.PhotoImage(current_image)
        
        
        image_label.configure(image=image_tk)
        image_label.image = image_tk
        
        #print(image_label.winfo_height()) =19
        #print(image_label.winfo_width()) =4
       
        
        classify_button.configure(state=tk.NORMAL)
        
def view_classified_images(list_classified, name):
    if list_classified:
        new_window = tk.Toplevel(window)
        new_window.title("Classified Images: " + str(name))
        new_window.geometry("800x600")
        
        row_count = 0
        col_count = 0
        for image_path in list_classified:
            classified_image = Image.open(image_path)
            classified_image = classified_image.resize((200, 150))
            classified_image_tk = ImageTk.PhotoImage(classified_image)
            
            image_label = tk.Label(new_window, image=classified_image_tk)
            
            image_label.image = classified_image_tk
            image_label.grid(row=row_count, column=col_count, padx=10, pady=10)
            
            col_count += 1
            if col_count > 3:
                col_count = 0
                row_count += 1
    else:
        tk.messagebox.showinfo("No Classified Images", "There are no classified images.")
        
def create_button_view(name_vehicle, classified_list, x, y):
    view_button = tk.Button(window, text=name_vehicle, command=lambda: view_classified_images(classified_list, name_vehicle),width=10,height=2)
    view_button.place(x=x, y=y)  

def vehicle_classification():
    global result
    global file_path
    global accuracy
    current_image_resized = load_img(file_path, target_size=(90, 120))
    current_image_array = img_to_array(current_image_resized)
    current_image_array = current_image_array.reshape(1, 90, 120, 3)
    current_image_array = current_image_array.astype('float32') / 255
    
    classified = model.predict(current_image_array)
    classified_class_index = np.argmax(classified)
    result = labels_vehicle[ classified_class_index]
    accuracy =classified[0][classified_class_index] * 100
    
    
    comment_textbox.delete(1.0, tk.END)
    comment_textbox.insert(tk.END, result)
    
    comment_textbox1.delete(1.0, tk.END)
    comment_textbox1.insert(tk.END, accuracy)
    
    if result == 'bicycle':
        classified_images_bicycle.append(file_path)
    elif result == 'boat':
        classified_images_boat.append(file_path)
    elif result == 'car':
        classified_images_car.append(file_path)
    elif result == 'motorbike':
        classified_images_motorbike.append(file_path)
    elif result == 'airplane':
        classified_images_airplane.append(file_path)
    elif result == 'train':
        classified_images_train.append(file_path)
    elif result == 'truck':
        classified_images_truck.append(file_path)

def open_enter(e):
    open_button.config(bg="red", fg="white")

def open_leave(e):
    open_button.config(bg="white", fg="black")
    
def classify_enter(e):
    classify_button.config(bg="red", fg="white")

def classify_leave(e):
    classify_button.config(bg="white", fg="black")

window = tk.Tk(screenName="VEHICLE")
window.geometry("600x500")
window.title("Vehicle Classification")

# canvas = tk.Canvas(window, width=600, height=600)
# canvas.place(x=0,y=0)
# canvas.create_rectangle(0, 0,600,600, fill="lightgray")
   
open_button = tk.Button(window, text="OPEN IMAGE", command=open_image,width=15,height=2)
open_button.place(x=10, y=10)
open_button.bind("<Enter>", open_enter)
open_button.bind("<Leave>", open_leave)

classify_button = tk.Button(window, text="CLASSIFICATION", command=vehicle_classification, state=tk.DISABLED,width=15,height=2)
classify_button.place(x=10, y=50)
classify_button.bind("<Enter>", classify_enter)
classify_button.bind("<Leave>", classify_leave)

#tạo nền cho các nút ấn
canvas = tk.Canvas(window, width=115, height=300)
canvas.place(x=10,y=130)
canvas.create_rectangle(0, 0, 115, 300, fill="#b2d8d8")

create_button_view('BICYCLE', classified_images_bicycle, 30, 140)
create_button_view('BOAT', classified_images_boat, 30, 180)
create_button_view('CAR', classified_images_car, 30, 220)
create_button_view('MOTORBIKE', classified_images_motorbike, 30, 260)
create_button_view('AIRPLANE', classified_images_airplane, 30, 300)
create_button_view('TRAIN', classified_images_train, 30, 340)
create_button_view('TRUCK', classified_images_truck, 30, 380)



#tạo nền cho hình ảnh vào giai đoạn đầu
canvas = tk.Canvas(window, width=420, height=420)
canvas.place(x=150,y=10)
canvas.create_rectangle(0, 0, 420, 420, fill="lightgray")

image_label = tk.Label(window)
image_label.place(x=160,y=20)

label = tk.Label(window, text="VEHICLE:")
label.place(x=170,y=330)

comment_textbox = tk.Text(window, width=20, height=1)
comment_textbox.place(x=250, y=330)

label = tk.Label(window, text="ACCURACY:")
label.place(x=170,y=370)

comment_textbox1 = tk.Text(window, width=20, height=1)
comment_textbox1.place(x=250, y=370)

window.mainloop()
