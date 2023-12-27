import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np 
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define a simple CNN model for image smoothing
class ScSpCNN(nn.Module):
    def __init__(self):
        super(ScSpCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        sLayer = self.decoder(x)
        dLayer = self.decoder2(x)
        out = torch.add(sLayer, dLayer)
        return sLayer, dLayer, out 
    
    
class ToGrayscale(object):
    def __call__(self, sample):
        grayscale_image = transforms.functional.to_grayscale(sample)
        return grayscale_image
    
# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 100

# Load and preprocess the CIFAR-100 dataset
transform = transforms.Compose([transforms.Resize((128, 128)),ToGrayscale(), transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = ScSpCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def my_l0(Y_hat,l):
    return l * torch.mean(torch.log(1 + torch.abs(Y_hat[:,:])))
def tv_loss_l1(Y_hat, l):
    return l * torch.mean(torch.abs(Y_hat[:,:, 1:] - Y_hat[:,:, :-1]))
def tv_loss_l2(Y_hat,l):
    return l * torch.mean(torch.square(Y_hat[:,:, 1:] - Y_hat[:,:, :-1]))


save_dir = 'ScSpNet_model'
LARGEFONT =("Verdana", 25)
  
class tkinterApp(tk.Tk):
    def __init__(self, *args, **kwargs): 
         
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("ScSpNET")
        self.geometry("1200x700")

        container = tk.Frame(self)  
        container.pack(side = "top", fill = "both", expand = True) 
  
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
  
        self.frames = {}  
  
        for F in (StartPage, Page1, Page2):
  
            frame = F(container, self)
  
            self.frames[F] = frame 
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(StartPage)
  
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
  
class StartPage(tk.Frame):
    
    def test(self):
        if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
           model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
           print('==> loading existing model:', os.path.join(save_dir, 'checkpoint.pth.tar'))
           model.load_state_dict(model_info['state_dict'])
           optimizer = torch.optim.Adam(model.parameters())
           optimizer.load_state_dict(model_info['optimizer'])
        else:
           print("Model is not saved")
        sample_image = Image.open(self.file_path)
        
        print(sample_image.mode)
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),  # Converts the image to a PyTorch tensor
        ])
        
        if(len(sample_image.mode) >= 3):
            if(len(sample_image.mode) == 4):
                r, g, b, a = sample_image.split()
            
            elif(len(sample_image.mode) == 3):
                r, g, b = sample_image.split()
                print(r.size)
                 
            # Apply the transformation to the image
            r = transform(r).unsqueeze(0).to(device)
            initial_r = r
            sLayer1, dLayer1, _ = model(r)
            
            g = transform(g).unsqueeze(0).to(device)
            sLayer2, dLayer2, _ = model(g)
            
            b = transform(b).unsqueeze(0).to(device)
            sLayer3, dLayer3, _ = model(b)
                            
            combined_sLayers = torch.stack([sLayer1, sLayer2, sLayer3], dim=1)
            combined_dLayers = torch.stack([dLayer1, dLayer2, dLayer3], dim=1)
            
            combined_sLayers = torch.clamp(combined_sLayers, 0, 1)
            combined_dLayers = torch.clamp(combined_dLayers, 0, 1)
            
            combined_sLayers = combined_sLayers.squeeze().detach().permute(1, 2, 0).cpu().numpy()
            combined_dLayers = combined_dLayers.squeeze().detach().permute(1, 2, 0).cpu().numpy()
            
            
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(sample_image)
            
            plt.subplot(1, 3, 2)
            plt.title("Smooth Layer")
            plt.imshow((combined_sLayers))
            plt.subplot(1, 3, 3)
            plt.title("Detail Layer")
            plt.imshow((combined_dLayers))
            
            file_path = "result.png"
            plt.savefig(file_path)
            
            image = Image.open(file_path)
            photo = ImageTk.PhotoImage(image)
            
            self.result_label.config(image=photo)
            self.result_label.photo = photo  
            
            try:
                line = int(self.input_entry.get())
                
                x = np.linspace(0, len(initial_r.squeeze().cpu().numpy()[line])-1, len(initial_r.squeeze().cpu().numpy())) 
                fig, ax = plt.subplots()
                plt.axis('off')
                ax = fig.add_subplot(5, 1, 1)
                ax.plot(x, (initial_r.squeeze().detach().cpu().numpy()[line] * 255).astype(np.uint8))
                ax.set_ylim([0,256])
                ax = fig.add_subplot(5, 1, 3)
                ax.plot(x, (sLayer1.squeeze().detach().cpu().numpy()[line] * 255).astype(np.uint8)) 
                ax.set_ylim([0,256])
                
                ax = fig.add_subplot(5, 1, 5)
                ax.plot(x, (dLayer1.squeeze().detach().cpu().numpy()[line] * 255).astype(np.uint8)) 
                ax.set_ylim([0,256])
                
                file_path = "result2.png"
                plt.savefig(file_path)
                
                image = Image.open(file_path)
                photo = ImageTk.PhotoImage(image)
                
                self.analyze_label = ttk.Label(self, text="")
                self.analyze_label.grid(row=5, column = 4)
                self.analyze_label.config(image=photo)
                self.test_button(text="Analyze single line of Image")
                self.analyze_label.photo = photo  # Store a
            
            except ValueError as e:
                # Code to handle the exception
                print(f"An exception of type {type(e).__name__} occurred: {str(e)}")
                
        else:
            sample_image = sample_image.convert("L")
            
            # Apply the transformation to the image
            sample_image = transform(sample_image).unsqueeze(0).to(device)
            sLayer, dLayer, output = model(sample_image)
            reconstructed_pil_image = transforms.ToPILImage()(sample_image.squeeze())
           
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(reconstructed_pil_image, cmap='gray')
            
            plt.subplot(1, 3, 2)
            plt.title("Smooth Layer")
            plt.imshow((sLayer.squeeze().detach().cpu().numpy()* 255).astype(np.uint8), cmap='gray')
            plt.subplot(1, 3, 3)
            plt.title("Detail Layer")
            plt.imshow((dLayer.squeeze().detach().cpu().numpy()* 255).astype(np.uint8), cmap='gray')
            
            file_path = "result.png"
            plt.savefig(file_path)
            
            image = Image.open(file_path)
            photo = ImageTk.PhotoImage(image)
            
            self.result_label.config(image=photo)
            self.result_label.photo = photo  
        
            try:
                line = int(self.input_entry.get())
                
                x = np.linspace(0, len(sample_image.squeeze().cpu().numpy()[line])-1, len(sample_image.squeeze().cpu().numpy())) 
                fig, ax = plt.subplots()
                plt.axis('off')
                ax = fig.add_subplot(5, 1, 1)
                ax.plot(x, (sample_image.squeeze().cpu().numpy()[line]* 255).astype(np.uint8))
                ax.set_ylim([0,256])
                ax = fig.add_subplot(5, 1, 3)
                ax.plot(x, (sLayer.squeeze().detach().cpu().numpy()[line] * 255).astype(np.uint8)) 
                ax.set_ylim([0,256])
                
                ax = fig.add_subplot(5, 1, 5)
                ax.plot(x, (dLayer.squeeze().detach().cpu().numpy()[line] * 255).astype(np.uint8)) 
                ax.set_ylim([0,256])
                
                file_path = "result2.png"
                plt.savefig(file_path)
                
                image = Image.open(file_path)
                photo = ImageTk.PhotoImage(image)
                
                self.analyze_label = ttk.Label(self, text="")
                self.analyze_label.grid(row=5, column = 4)
                self.analyze_label.config(image=photo)
                self.test_button(text="Analyze single line of Image")
                self.analyze_label.photo = photo  # Store a
            
            except ValueError as e:
                # Code to handle the exception
                print(f"An exception of type {type(e).__name__} occurred: {str(e)}")
 
    def open_image_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp")])
        if file_path:
            self.file_path = file_path
            image = Image.open(file_path)
            photo = ImageTk.PhotoImage(image)
    
            self.image_label.config(image=photo)
            self.image_label.photo = photo  # Store a reference to prevent it from being garbage collected
            
            self.test_button = ttk.Button(self, text="Test Image", command= self.test)
    
            self.label2 = ttk.Label(self, text="Enter a line to analyze: ")
            self.label2.grid(row = 3, column=3)

            self.input_entry = ttk.Entry(self)
            self.input_entry.grid(row = 3 , column= 4)

            self.result_label2 = ttk.Label(self, text="")
            self.result_label2.grid(row = 4, column = 4)
            self.test_button.grid(row = 3, column = 5)

                
    def __init__(self, parent, controller): 
        tk.Frame.__init__(self, parent)
        self.file_path = ""
        label = ttk.Label(self, text ="Test Single Image", font = LARGEFONT)
         

        label.grid(row = 0, column = 4, padx = 10, pady = 10) 
  
        button1 = ttk.Button(self, text ="Train Your Model",
        command = lambda : controller.show_frame(Page1))
     
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)
  
        button2 = ttk.Button(self, text ="Test Multiple Image",
        command = lambda : controller.show_frame(Page2))
     
        button2.grid(row = 2, column = 1, padx = 10, pady = 10)
  
        self.image_label = ttk.Label(self)
        self.image_label.grid(row = 1, column = 4)

        open_button = ttk.Button(self, text="Open Image", command= self.open_image_dialog)
        open_button.grid(row = 2, column = 4)
        
        self.test_button = ttk.Button(self, text="Test Image", command= self.test)

        self.result_label = ttk.Label(self, text="")
        self.result_label.grid(row = 1, column = 4)
        
        self.label2 = ttk.Label(self, text="")
        self.input_entry = ttk.Entry(self)
        
        self.result_label2 = ttk.Label(self, text="")
        self.result_label2.grid(row = 4, column = 4)
        
        

# Create a custom dataset that loads all images from a directory
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image

# Train Your Model
class Page1(tk.Frame):
     
    def select_folder_dialog(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            # Hyperparameters
            batch_size = 32
            learning_rate = 0.001
            num_epochs = 100

            transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
            train_dataset = CustomImageDataset(root_dir=folder_path, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  
    
            # Initialize the model, loss function, and optimizer
            model = ScSpCNN().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
            def my_l0(Y_hat,l):
                return l * torch.mean(torch.log(1 + torch.abs(Y_hat[:,:])))
            def tv_loss_l1(Y_hat, l):
                return l * torch.mean(torch.abs(Y_hat[:,:, 1:] - Y_hat[:,:, :-1]))
    
            def tv_loss_l2(Y_hat,l):
                return l * torch.mean(torch.square(Y_hat[:,:, 1:] - Y_hat[:,:, :-1]))
    

            save_dir = 'my_model'
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
                
            # Training loop
            for epoch in range(num_epochs):
                for batch_data in train_loader:
                    inputs = batch_data
                    noisy_inputs = inputs.to(device) + 0.1 * torch.randn(inputs.size()).to(device)  # Add some artificial noise (adjust the scale as needed)
                    noisy_inputs = torch.clamp(noisy_inputs, 0, 1)  # Add some artificial noise (adjust the scale as needed)
                    optimizer.zero_grad()
                    sLayers, dLayers, outputs = model(noisy_inputs.to(device))
                    tvLoss = tv_loss_l2(sLayers,0.50) + tv_loss_l1(dLayers,0.002) + my_l0(dLayers, 0.002)
                    loss = criterion(outputs, noisy_inputs.to(device)) + tvLoss
                    loss.backward()
                    optimizer.step()
                    torch.save({
             			'epoch': epoch + 1,
             			'state_dict': model.state_dict(),
             			'optimizer' : optimizer.state_dict()},
             			os.path.join(save_dir, 'checkpoint.pth.tar'))
                    
                self.result_text.insert(tk.END, f"Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item()} TV Loss: {tvLoss.item()}\n")
                self.result_text.yview(tk.END)
                app.update_idletasks()
        self.result_text.insert(tk.END, "Your model is saved under " + save_dir + " directory \n")
        self.result_text.insert(tk.END, "If you change checkpoint file with the checkpoint file in ScSpNET_model file, \n")
        self.result_text.insert(tk.END, "you can test your own model! \n")
        self.result_text.yview(tk.END)
        app.update_idletasks()
    def __init__(self, parent, controller):
         
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text ="Train Your Model", font = LARGEFONT)
        label.grid(row = 1, column = 0, padx = 10, pady = 10)
  
        button1 = ttk.Button(self, text ="Test Single Image",
                            command = lambda : controller.show_frame(StartPage))
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)
        
        button2 = ttk.Button(self, text ="Test Multiple Image",
                            command = lambda : controller.show_frame(Page2))
     
        button2.grid(row = 1, column = 2, padx = 10, pady = 10)
        #button2.grid(row = 2, column = 1, padx = 10, pady = 10)
        
        selectFolder_button = ttk.Button(self, text="Select training images folder", command= self.select_folder_dialog)
        selectFolder_button.grid(row = 2, column = 0)
        
        self.result_text = tk.Text(self, height=20, width=80)
        self.result_text.grid(row = 3, column = 0, padx=10, pady=10, sticky="nsew")
        self.result_text.config(wrap=tk.WORD)
        
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.result_text.yview)
        scrollbar.grid(row=3, column=1, sticky="ns")
        self.result_text.config(yscrollcommand=scrollbar.set)

        
  
# Test Multiple Images
class Page2(tk.Frame): 
    
    def testImage(self, sample_image, file_name):
        if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
           model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
           print('==> loading existing model:', os.path.join(save_dir, 'checkpoint.pth.tar'))
           model.load_state_dict(model_info['state_dict'])
           optimizer = torch.optim.Adam(model.parameters())
           optimizer.load_state_dict(model_info['optimizer'])
        else:
           print("Model is not saved")
        
        # Define a transformation to convert the PIL image into a tensor
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts the image to a PyTorch tensor
        ])
        sample_image = sample_image.resize((128, 128))  # Resize the image if needed
        
        if(len(sample_image.mode) >= 3):
            if(len(sample_image.mode) == 4):
                r, g, b, a = sample_image.split()
            
            elif(len(sample_image.mode) == 3):
                r, g, b = sample_image.split()
                print(r.size)
                 
            # Apply the transformation to the image
            r = transform(r).unsqueeze(0).to(device)
            sLayer1, dLayer1, _ = model(r)
            
            g = transform(g).unsqueeze(0).to(device)
            sLayer2, dLayer2, _ = model(g)
            
            b = transform(b).unsqueeze(0).to(device)
            sLayer3, dLayer3, _ = model(b)
                            
            combined_sLayers = torch.stack([sLayer1, sLayer2, sLayer3], dim=1)
            combined_dLayers = torch.stack([dLayer1, dLayer2, dLayer3], dim=1)
            
            combined_sLayers = torch.clamp(combined_sLayers, 0, 1)
            combined_dLayers = torch.clamp(combined_dLayers, 0, 1)
            
            sLayer = combined_sLayers.squeeze().detach().permute(1, 2, 0).cpu().numpy()
            dLayer = combined_dLayers.squeeze().detach().permute(1, 2, 0).cpu().numpy()
            
            saveSmoothDir = "smoothResults"
            if not os.path.isdir(saveSmoothDir):
                os.makedirs(saveSmoothDir)
                
            saveDetailDir = "detailResults"
            if not os.path.isdir(saveDetailDir):
                os.makedirs(saveDetailDir)  
            
            image_array = (sLayer * 255).astype(np.uint8)

            # Convert the NumPy array to a PIL Image
            image_pil = Image.fromarray(image_array)
            image_pil.save("./" + saveSmoothDir +"/"+file_name)
            
            image_array = (dLayer * 255).astype(np.uint8)
            
            # Convert the NumPy array to a PIL Image
            image_pil = Image.fromarray(image_array)
            image_pil.save("./" + saveDetailDir +"/"+file_name)
            
        else:
            sample_image = sample_image.convert("L")
            # Apply the transformation to the image
            sample_image = transform(sample_image).unsqueeze(0).to(device)
            sLayer, dLayer, output = model(sample_image)
            
            sLayer = sLayer.squeeze().detach().cpu().numpy();
            dLayer = dLayer.squeeze().detach().cpu().numpy()
            
            saveSmoothDir = "smoothResults"
            if not os.path.isdir(saveSmoothDir):
                os.makedirs(saveSmoothDir)
                
            saveDetailDir = "detailResults"
            if not os.path.isdir(saveDetailDir):
                os.makedirs(saveDetailDir)  
            image = Image.fromarray((sLayer * 255).astype(np.uint8))
            image.save("./" + saveSmoothDir +"/"+file_name)
            
            image = Image.fromarray((dLayer * 255).astype(np.uint8))
            image.save("./" + saveDetailDir + "/"+file_name)
        
    def select_folder_dialog(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            training_start_time = time.time()
            for filename in os.listdir(folder_path):
                if filename.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                    image_path = os.path.join(folder_path, filename)
                    image = Image.open(image_path)
                    self.testImage(image, filename)
            print('Training finished, took {:.2f}'.format(time.time() - training_start_time))   
            
            
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        label = ttk.Label(self, text ="Test Multiple Image", font = LARGEFONT)
        label.grid(row = 0, column = 4, padx = 10, pady = 10)
  

        button1 = ttk.Button(self, text ="Train Your model",
                            command = lambda : controller.show_frame(Page1))

        button1.grid(row = 1, column = 1, padx = 10, pady = 10)
    
        button2 = ttk.Button(self, text ="Test Single Image",
                            command = lambda : controller.show_frame(StartPage))
    
        button2.grid(row = 1, column = 2, padx = 10, pady = 10)
        selectFolder_button = ttk.Button(self, text="Select input images folder", command= self.select_folder_dialog)
        selectFolder_button.grid(row = 1, column = 4)

# Driver Code
app = tkinterApp()
app.title = "ScSpNET"
app.mainloop()