# ScSpNET
An AutoEncoder Model to divide an image into two scales of information

ScSpNET is a Python based unsupervised algorithm designed for the partitioning 2D images into two distinct information scales: the smooth and detail scales. First layer presents a smoothed version of the input image, while the second layer reveals finer details using a simple convolutional neural network model. The algorithm is provided by a user-friendly graphical interface, providing options for testing single-image with detailed analysis or testing multiple image with a pretrained model. Additionally, users can have the capability to train proposed model with their own datasets by using GUI and they have the flexibility to extend the software's functionality by modifying the source code.


### Used Packages

#### Name       -          Version                   
matplotlib                3.7.2          
numpy                     1.25.2          
opencv                    4.6.0          
pillow                    9.4.0          
python                    3.11.5              
pytorch                   2.0.1          
pytorch-cuda              11.8              
scikit-image              0.20.0        
scipy                     1.11.1          
tk                        8.6.12               
torchvision               0.15.2               
