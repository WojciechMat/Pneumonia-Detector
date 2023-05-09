# Pneumonia Detection
### CAD
*Computer-aided diagnosis* (CAD) is a field of medicine that uses artificial intelligence (AI) and machine learning algorithms to assist in the diagnosis of medical conditions. CAD systems analyze medical images, patient data, and other relevant information to help doctors make more accurate and efficient diagnoses. <br>
Thanks to machine learning we can analyze large datasets of biological information to draw conclusions.
I believe it is not possible to replace a radiologist in diagnostic process but CAD can help doctors in making a decision or draw their attention to details that they might have missed.
### Extraction of information
Currently, the information that is being feeded to a model is an *700px by 700px*, black and white image passed as *(300, 300, 1)* - shaped tensor. This is very unefficient way of obtaining information from lung x-ray images. The model accuracy would be much better if I were to segment lungs and delete objects from scans. The tool I can use is [lung-segmentation-2d](https://github.com/imlab-uiip/lung-segmentation-2d) from user [imlab-uiip](https://github.com/imlab-uiip). I am planning on implementing this feature in the future.
### Model
This model is a convulsion neural network mixing convulsion layers with average and max pooling, finished with Dense layers. It has the accuracy of **~80%** which may somewhat help in making a final decision by a radiologist but is rather unsatisfactory. It does however reach a 94.4% TP accuracy. <br>
 <img src=https://github.com/WojciechMat/Pneumonia-Detector/blob/main/plots/confusion_matrix.png width="500" height="500">
<br>

There is still a lot of room for development.
