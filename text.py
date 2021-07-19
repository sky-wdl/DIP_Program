def openimage(self):



    open_image = tk.Tk
    open_image().withdraw()
    open_image_path = filedialog.askopenfile()
    print(open_image_path.name)
