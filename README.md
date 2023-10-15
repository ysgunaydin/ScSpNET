# ScSpNET
An AutoEncoder Model to divide an image into two scales of information

ScSpNET is a Python based unsupervised algorithm designed for the partitioning 2D images into two distinct information scales: the smooth and detail scales. First layer presents a smoothed version of the input image, while the second layer reveals finer details using a simple convolutional neural network model. The algorithm is provided by a user-friendly graphical interface, providing options for testing single-image with detailed analysis or testing multiple image with a pretrained model. Additionally, users can have the capability to train proposed model with their own datasets by using GUI and they have the flexibility to extend the software's functionality by modifying the source code.

# Installation of ScSpNET

## Python Installation

- You can download python from the official web page of Python https://www.python.org/downloads/ according to your Operating System.
- python 3.11.5 is used for developing the software, this version and later versions can be preferred.
- After you install you can open **Command Line** from your computer and check its version by typing the following command.

 <pre>   
   python --version
 </pre>

 - It will show you the version of your python like following. If you are not seeing the version of Python please turn back to the installation step.
  <pre>   
   Python 3.12.0
  </pre>

 - After you install python you need to install the following packages by using either pip or pip3 command on your command line according to your python version.

  <pre>   
    pip3 install matplotlib
    pip3 install numpy
    pip3 install opencv
    pip3 install pillow 
    pip3 install pytorch
    pip3 install scikit-image
    pip3 install scipy
    pip3 install tk
    pip3 install torchvision
  </pre>

- Then you can go to the directory of ScSpNET.py file on your command line and run following command:
<pre>   
    python ScSpNET.py
</pre>

- Finally you will see a graphical user interface of the software and you are able to use it in your computer.


- If you don't want to handle these processes or if you stuck at some point, there is an executable version of the program (3GB) and you can request the executable file of the software from: yusufsevkignydn@gmail.com.

The list of used packages are given below.

### Used Packages

#### Name       -          Version                   
matplotlib                3.7.2          
numpy                     1.25.2          
opencv                    4.6.0          
pillow                    9.4.0                      
pytorch                   2.0.1          
scikit-image              0.20.0        
scipy                     1.11.1          
tk                        8.6.12               
torchvision               0.15.2               
