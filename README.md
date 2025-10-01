
# Artificial Intelligence Data Science: Boat Classification with Deep Learning

# Project Background:
To decrease human error for boat classification, Marina Pier Inc. automates its port operations Hand-operated misclassification of boats has caused inefficient work. To handle this problem, the company is implementing deep learning models. The goal is to make a self-operating system which can precisely categorize various boat types. Furthermore, a lightweight transfer learning model is engineered to support implementation on mobile devices. 

# Tools and Technologies:
•	Programming language (Python)
•	Libraries (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow, Keras)
•	Deep learning models used (CNN and MobileNetV2)
•	Techniques used: preprocessing (image), scaling (data), analysis (confusion matrix), , early stopping, precision-recall evaluation, and transfer learning. .

# Workflow:
•	Data Preparation (Used a dataset of 1,162 images from 9 boat categories) with splitting   dataset into training, validation, and testing sets including normalization 
•	Model Development (CNN): Make a convolutional neural network with Conv2D, MaxPooling, GlobalAveragePooling, Dense, and Softmax layers. Trained the model for 20 epochs and evaluated performance using accuracy, precision, and recall. Overview dataset training plots, heatmap for confusion matrix, and classification results
•	Model Development: Initialized MobileNetV2 as the base model. Added other layers with dropout. Used batch normalization. Trained for 50 epochs with early stopping on validation loss. Evaluated test performance and plotted accuracy vs. loss curves.
•	Launch maturity: Designed lightweight framework fully used for mobile devices. Achieved unbiased, mechanized classification system matched with functional goals. 

# Objectives: 
•	use AI with deep learning methods for automate port operations. 
•	Make a Convolutional Neural Network (CNN) model 
•	Train the CNN model to classify test images.
•	implement a MobileNetV2 model to classify test images.
•	use performance metrics such as confusion matrix, classification report, accuracy, loss, precision, and recall for models’ evaluation.
•	Prepare the trained models for possible application on various capacity-limited devices such as mobile and hardware. 

# Conclusions:
•	The project successfully showed the application of deep learning for automating image detection in port operations.
•	A CNN model was able to learn basic understandings but struggled to get high accuracy because of limited dataset size and sophistication.
•	The MobileNetV2 model showed higher validation performance, highlighting the strength of pretrained models on small datasets.
•	Evaluation metrics showed class imbalance and data limitations which are key hurdles that affected the foundational CNN.
•	Transfer learning improved accuracy as well as offered better generalization and stability for unseen test data.
•	The overall workflow established a foundation for scaling AI solutions used in industrial port automation, and transfer learning being the recommended approach for deployment.

# jupyter notebook file:  Artificial_Intelligence_Data Science_Deep_Learning_Automating_Port_Operations_Keras_Tensorflow.ipynb
