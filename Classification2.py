import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score
from keras_tuner import RandomSearch, Hyperband
import tensorflow_addons as tfa
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import keras_tuner as kt
import random  
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

#Balanced Accuracy for plotting. 
def BalancedAccuracy(y_true, y_pred):
    
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    
    num_classes = 6
    
    # Calculate the confusion matrix
    conf_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)
    
    # Initialize variables to store the values for each class
    TP = [0] * num_classes
    TN = [0] * num_classes
    FP = [0] * num_classes
    FN = [0] * num_classes
    
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                TP[i] = conf_matrix[i, i]
            else:
                FP[i] += conf_matrix[i, j]
                FN[j] += conf_matrix[i, j]
    
    # Calculate specificity and sensitivity for each class
    specificity = [0.0] * num_classes
    sensitivity = [0.0] * num_classes
    
    for i in range(num_classes):
        C0 = TN[i] + FP[i]
        C1 = TP[i] + FN[i]
        if C0 != 0:
            specificity[i] = float(TN[i]) / float(C0)
        if C1 != 0:
            sensitivity[i] = float(TP[i]) / float(C1)
    
    # Calculate balanced accuracy as the average of sensitivity and specificity
    BalancedAccuracy = (sum(specificity) + sum(sensitivity)) / 6
    
    return BalancedAccuracy

#Tests the one format that we achieved better results on first classification problem.
def smote_function(X_train_subset, y_train_subset):
    
    smote = SMOTE(sampling_strategy = 'auto', k_neighbors = 10)
    x_train_reshaped, y_train_reshaped = smote.fit_resample(X_train_subset, y_train_subset)

    x_train_reshaped = X_train_subset.reshape(-1,28,28,3)
    y_train_reshaped = to_categorical(y_train_subset)

    return x_train_reshaped, y_train_reshaped

#Starting the convolutional neural network
def network():
                         # Layer by Layer  we start to build the neural network #
    
    # Initialize network in a squential manner.
    model = tf.keras.models.Sequential()
    
        #Data augmentation layer - generalize better without introducing overly aggressive distortions that might hinder learning#
                                    
    #We randomly flip the input horizontally and vertically and rotate it up to 0.5 radians.
    model.add(tf.keras.layers.RandomFlip("horizontal_and_vertical"))  
    model.add(tf.keras.layers.RandomRotation(0.5))
    # model.add(tf.keras.layers.RandomZoom(0.5))
    # model.add(tf.keras.layers.RandomContrast(0.5))

                        #1st convolutional layer followed by a max pooling layer (reduces the spatial dimensions by a factor of 3x3)#
                        
    #The convolutional layer uses 32 filters with a 3x3 kernel, ReLU activation, and expects input with a shape of (28, 28, 3)                  
    model.add(tf.keras.layers.Conv2D(32, kernel_size=2, activation='relu', input_shape=(28,28,3)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2)) 
    
                        #2nd convolutional layer followed by a max pooling layer#
    
    #Same configuration as the first one.
    model.add(tf.keras.layers.Conv2D(32, kernel_size=2, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

    
    ## 3rd convolutional layer
    model.add(tf.keras.layers.Conv2D(32, kernel_size=2, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
    
                            #Fully connected classifier#
    
    #convert the 2D feature maps from the convolutional layers into a 1D vector, 
    #which can be fed into fully connected layers
    model.add(tf.keras.layers.Flatten())
 
                        #1st dense (fully connected) layer#
                        
    model.add(tf.keras.layers.Dense(1024, activation='relu')) 
    #Dropout is a regularization technique that helps prevent overfitting. 
    #It randomly deactivates 50% (specified by the 0.5) of the neurons during each training iteration, forcing the network to learn more robust and generalized representations.
    model.add(tf.keras.layers.Dropout(0.5))     

                        #2st dense (fully connected) layer#
                        
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))    

                        #3st dense (fully connected) layer#
                        
    model.add(tf.keras.layers.Dense(56, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))    
    
    #The use of three dense layers in the script allows the neural network to learn increasingly complex
    #and abstract representations from the input data.
       
                                  #Output layer#
    
    #The final dense layer with 6 units and softmax activation is used for binary classification. 
    #This layer will produce probability scores for the six classes.
    model.add(tf.keras.layers.Dense(6, activation ='softmax'))
    #model.summary()

                                    #Compile#
                        
    #The model is compiled with the Adam optimizer, a binary cross-entropy loss function, and accuracy 
    #and Balanced Accuracy as evaluation metrics. The learning rate for the optimizer is set to 0.0002 (The smaller the better).                  
    model.compile(tf.keras.optimizers.Adam(0.0002), loss='categorical_crossentropy' ,  #loss=tf.keras.losses.BinaryCrossentropy(),  #loss='categorical_crossentropy'
                  metrics=['accuracy', BalancedAccuracy])  #, BalancedAccuracy #tf.keras.metrics.CategoricalAccuracy(name='accuracy')
    
    return model

#Defines the number of samples of each class
def classLabel(cardinal, y):
    
    Total_samples = y.shape[0]
    Number_of_samples = len([cardinal for i in range(Total_samples) if y[i] == cardinal])
    
    return Number_of_samples

#Counts the number of samples from Dermoscopy DataSet. 
def countDermoscopySamples(y):
    
    nevu_class = 0
    melanoma_class = 1
    vascular_lesions_class = 2
    
    num_nevu = classLabel(nevu_class, y)
    num_melanoma = classLabel(melanoma_class, y)
    num_vascular_lesions = classLabel(vascular_lesions_class, y)

    total = num_nevu + num_melanoma + num_vascular_lesions
    
    return num_nevu, num_melanoma, num_vascular_lesions, total

#Counts the number of samples from Blood Cell DataSet. 
def countBloodCellSamples(y): 

    granulocytes_class = 3
    basophils_class = 4
    lymphocytes_class = 5

    num_granulocytes = classLabel(granulocytes_class, y)
    num_basophils = classLabel(basophils_class, y)
    num_lymphocytes = classLabel(lymphocytes_class, y)
    
    total = num_granulocytes + num_basophils + num_lymphocytes

    return num_granulocytes, num_basophils, num_lymphocytes, total

#Counts the total number of samples , and the number of samples of each data set.
def countSamples(y):
    Dermoscopy = countDermoscopySamples(y)
    BloodCell = countBloodCellSamples(y)
    
    DermoscopySet = Dermoscopy[3]
    BloodCellSet = BloodCell[3]

    total = DermoscopySet + BloodCellSet
    return total, DermoscopySet, BloodCellSet

#Loads the data
def LoadData():
    
    # Loading data from the given files
    Xtrain = np.load('Xtrain_Classification2.npy')
    ytrain = np.load('ytrain_Classification2.npy')
    Xtest = np.load('Xtest_Classification2.npy')
    
    return Xtrain, ytrain, Xtest

#When the data is subsampled (used initially but then removed)
def subsample(x, y):
    
    y = np.array(y)
    x = np.array(x)

    if isinstance(y[0], (int, float)) and y[0] not in [3, 4, 5]:

        labels = y
        #number of each data set
        Dermoscopic = countDermoscopySamples(y)
        
        #The number of samples of each class
        N_nevu = Dermoscopic[0]
        N_melanoma = Dermoscopic[1]
        N_vascular_lesions = Dermoscopic[2]

        #finds the class with less samples
        min_samples = min(N_nevu, N_melanoma, N_vascular_lesions)
        if min_samples == N_nevu:
            minimum_class = 0
        elif min_samples == N_melanoma:
            minimum_class = 1
        elif min_samples == N_vascular_lesions:
            minimum_class = 2

        print("minimum_class", minimum_class)
        print("min_samples", min_samples)

        #finds the class with more samples
        max_samples = max(N_nevu, N_melanoma, N_vascular_lesions)
        if max_samples == N_nevu:
            max_class = 0
        elif max_samples == N_melanoma:
            max_class = 1
        elif max_samples == N_vascular_lesions:
            max_class = 2
        
        #finds the class with the intermediate number of samples
        if N_nevu != min_samples and N_nevu != max_samples:
            medium_samples = N_nevu
            medium_class = 0
        elif N_melanoma != min_samples and N_melanoma != max_samples:
            medium_samples = N_melanoma
            medium_class = 1    
        elif N_vascular_lesions != min_samples and N_vascular_lesions != max_samples:
            medium_samples = N_vascular_lesions
            medium_class = 2


        N_class_to_remove1 = max_samples - min_samples 
        N_class_to_remove2 = medium_samples - min_samples

        LastIndex_max = max_samples - N_class_to_remove1
        LastIndex_medium = medium_samples - N_class_to_remove2
        
        #Working on max class#
        New_max = np.array([x[i, :] for i in range(len(labels)) if labels[i] == max_class])
        
        #Working on Max class witouth the excedentarys#
        New_max = New_max[:LastIndex_max, :]
        N_new_max = New_max.shape[0]
        New_max_label = [max_class for i in range(N_new_max)]

        #Working on medium class#   
        New_medium = np.array([x[i, :] for i in range(len(labels)) if labels[i] == medium_class])

        #Working on medium class witouth the excedentarys#
        New_medium = New_medium[:LastIndex_medium, :]
        N_new_medium = New_medium.shape[0]
        New_medium_label = [medium_class for i in range(N_new_medium)]

        #Working on Minimum class#
        minimum = np.array([x[i, :] for i in range(len(labels)) if labels[i] == minimum_class])
        minimum_label = [minimum_class for i in range(minimum.shape[0])]
        
        #concatenate all the classes
        conc_values = np.concatenate((np.array(New_max), np.array(New_medium), np.array(minimum)), axis = 0)
        conc_labels = np.concatenate((np.array(New_max_label), np.array(New_medium_label), np.array(minimum_label)))
        
        #shuffle
        mix = np.random.permutation(len(conc_labels))
        yfinal = conc_labels[mix]
        Xfinal = conc_values[mix].reshape(-1,28,28,3)
        
        #One-Hot Encoding# 
        yfinal = to_categorical(yfinal)
    
    else:
        
        BloodCell = countBloodCellSamples(y)

        N_granulocytes = BloodCell[0]
        N_basophils = BloodCell[1]
        N_lymphocytes = BloodCell[2]

        labels = y

        #finds the class with less samples
        min_samples = min(N_granulocytes, N_basophils, N_lymphocytes)
        
        if min_samples == N_granulocytes:
            minimum_class = 3
        elif min_samples == N_basophils:
            minimum_class = 4
        elif min_samples == N_lymphocytes:
            minimum_class = 5

        print("minimum_class", minimum_class)
        print("min_samples", min_samples)

        #finds the class with more samples
        max_samples = max(N_granulocytes, N_basophils, N_lymphocytes)
        
        if max_samples == N_granulocytes:
            max_class = 3
        elif max_samples == N_basophils:
            max_class = 4
        elif max_samples == N_lymphocytes:
            max_class = 5
        
        #finds the class with the intermediate number of samples
        if N_granulocytes != min_samples and N_granulocytes != max_samples:
            medium_samples = N_granulocytes
            medium_class = 3
        elif N_basophils != min_samples and N_basophils != max_samples:
            medium_samples = N_basophils
            medium_class = 4    
        elif N_lymphocytes != min_samples and N_lymphocytes != max_samples:
            medium_samples = N_lymphocytes
            medium_class = 5

        N_class_to_remove1 = max_samples - min_samples 
        N_class_to_remove2 = medium_samples - min_samples

        LastIndex_max = max_samples - N_class_to_remove1
        LastIndex_medium = medium_samples - N_class_to_remove2
        
        #Working on max class#
        New_max = np.array([x[i, :] for i in range(len(labels)) if labels[i] == max_class])
        
        #Working on Max class witouth the excedentarys#
        New_max = New_max[:LastIndex_max, :]
        N_new_max = New_max.shape[0]
        New_max_label = [max_class for i in range(N_new_max)]

        #Working on medium class#   
        New_medium = np.array([x[i, :] for i in range(len(labels)) if labels[i] == medium_class])

        #Working on medium class witouth the excedentarys#
        New_medium = New_medium[:LastIndex_medium, :]
        N_new_medium = New_medium.shape[0]
        New_medium_label = [medium_class for i in range(N_new_medium)]

        #Working on Minimum class#
        minimum = np.array([x[i, :] for i in range(len(labels)) if labels[i] == minimum_class])
        minimum_label = [minimum_class for i in range(minimum.shape[0])]
        
        #concatenate all the classes
        conc_values = np.concatenate((np.array(New_max), np.array(New_medium), np.array(minimum)), axis = 0)
        conc_labels = np.concatenate((np.array(New_max_label), np.array(New_medium_label), np.array(minimum_label)))
        
        #shuffle
        mix = np.random.permutation(len(conc_labels))
        yfinal = conc_labels[mix]
        Xfinal = conc_values[mix].reshape(-1,28,28,3)
        
        #One-Hot Encoding# 
        yfinal = to_categorical(yfinal)
    
    return Xfinal, yfinal

#Splits the data into two arrays, one for each dataset.
def separateData(x, y):
    
    values = countSamples(y)
    
    total_samples = values[0]
    
    DermoscopyArray = []
    BloodCellArray = []
    
    for i in range(total_samples):
        
        if y[i] == 0 or y[i] == 1 or y[i] == 2:
            DermoscopyArray.append((x[i], y[i]))
            
        elif y[i] == 3 or y[i] == 4 or y[i] == 5:
            BloodCellArray.append((x[i], y[i]))
            
    return DermoscopyArray, BloodCellArray

#Calculates the weight for each class.
def CalcWeight(class_, total_samples, number_classes):

    weight = (1 / class_) * (total_samples / number_classes)

    return weight    

#Plots and saves what we want.
def Plot_and_Save():
     
    # Customize the font size for labels
    label_fontsize = 12
    legend_fontsize = 6
    tick_label_fontsize = 8

    # Set the figure size
    plt.figure(1, figsize=(11, 3))
    
    # First plot
    plt.subplot(1, 3, 1)
    plt.xlabel(r"Epoch", fontsize=label_fontsize)
    plt.ylabel(r"Loss", fontsize=label_fontsize)
    plt.plot(history.history["loss"], label="training set", color="green")
    plt.plot(history.history["val_loss"], label="validation set", color="black")
    plt.legend(fontsize=legend_fontsize)

    # Second plot
    plt.subplot(1, 3, 2)
    plt.xlabel(r"Epoch", fontsize=label_fontsize)
    plt.ylabel(r"Accuracy", fontsize=label_fontsize)
    plt.plot(history.history["accuracy"], label="training set", color="blue")
    plt.plot(history.history["val_accuracy"], label="validation set", color="red")
    plt.legend(fontsize=legend_fontsize)

    # Third plot
    plt.subplot(1, 3, 3)
    plt.xlabel(r"Epoch", fontsize=label_fontsize)
    plt.ylabel(r"Balanced Accuracy", fontsize=label_fontsize)
    plt.plot(history.history["BalancedAccuracy"], label="training set", color="orange")
    plt.plot(history.history["val_BalancedAccuracy"],label="validation set", color="red")
    plt.legend(fontsize=legend_fontsize)

    # Adjust tick label font size
    for ax in plt.gcf().get_axes():
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(tick_label_fontsize)
    
    # Adjust subplot layout to prevent overlapping labels
    plt.subplots_adjust(wspace=0.4)  # Adjust the horizontal space between subplots

    # Save the plots as PNG files
    plt.savefig(f"plots_{BalAccuracy}.png", dpi=300, bbox_inches="tight")

    # Display the plots.
    plt.show()

    plt.close()
    
                                                #flags#
#In order to know if we are in first iteration 
first_iteration = 1

#training with class weight 
classWeights = 1

#The one we got better results (1 - used)
smote = 1

#not used
subsample = 0

#not used
oversample=0

#not used
equalize_data = 0

#For plotting the graphics in need (1 - They are plotted, 0 - not plotted)
show_plots = 1

# Read data
X_train, y_train, X_test = LoadData()

#Normalization 8-bit, 
#the original pixel values in an 8-bit image typically range from 0 (black) to 255 (white).
X_train = X_train / 255 
X_test = X_test / 255
#Now the scale is between [0, 1].

#If training with class_weights is activated.
if classWeights:
        
        #total samples from the Dermoscopy dataset.
        Dermoscopy = countDermoscopySamples(y_train)
        #total samples from the BloodCell dataset.
        BloodCell = countBloodCellSamples(y_train)
        
        #extracting the values to each class from Dermoscopy dataset.
        nevu = Dermoscopy[0]
        melanoma = Dermoscopy[1]
        vascular_lesions = Dermoscopy[2]
        
        #extracting the values to each class from BloodCell dataset.
        granulocytes = BloodCell[0]
        basophils = BloodCell[1]
        lymphocytes = BloodCell[2]
        
        #total samples.
        DataSamples = countSamples(y_train)
        #extracting the total samples.
        total_samples = DataSamples[0]
        
        #Calculating the weight for each class.
        weight_nevu = CalcWeight(nevu, total_samples, 6.0)
        weight_melanoma = CalcWeight(melanoma, total_samples, 6.0)
        weight_vascular_lesions = CalcWeight(vascular_lesions, total_samples, 6.0)
        weight_granulocytes = CalcWeight(granulocytes, total_samples, 6.0)
        weight_basophils = CalcWeight(basophils, total_samples, 6.0)
        weight_lymphocytes = CalcWeight(lymphocytes, total_samples, 6.0)
        
        #Assigning an weight for each class
        class_weights = {0: weight_nevu, 
                        1: weight_melanoma,
                        2: weight_vascular_lesions,
                        3: weight_granulocytes,
                        4: weight_basophils,
                        5: weight_lymphocytes}

#Splitting the data of the both data sets.
SplitData = separateData(X_train, y_train)

#Data correspondent to Dermosocpy DataSet 
train_Dermoscopy = SplitData[0]
#Data correspondent to BloodCell DataSet 
train_BloodCell = SplitData[1]

#get each column of the tuple
train_Dermoscopy_x = [train_Dermoscopy[i][0] for i in range(len(train_Dermoscopy))]
train_Dermoscopy_y = [train_Dermoscopy[i][1] for i in range(len(train_Dermoscopy))]
train_BloodCell_x = [train_BloodCell[i][0] for i in range(len(train_BloodCell))]
train_BloodCell_y = [train_BloodCell[i][1] for i in range(len(train_BloodCell))]

#The values are as List, but we want them as arrays.
train_Dermoscopy_x = np.array(train_Dermoscopy_x)
train_Dermoscopy_y = np.array(train_Dermoscopy_y)
train_BloodCell_x = np.array(train_BloodCell_x)
train_BloodCell_y = np.array(train_BloodCell_y)

#Get validation set & Test set (15% for validation set - 85% for test set) for Dermosocpy dataset    
Dermoscopy_subsetX, Dermoscopy_ValidX, Dermoscopy_subsetY, Dermoscopy_ValidY = train_test_split(train_Dermoscopy_x, train_Dermoscopy_y, train_size = 0.85)
#Get validation set & Test set (15% for validation set - 85% for test set) for BloodCell dataset        
BloodCell_subsetX, BloodCell_ValidX, BloodCell_subsetY, BloodCell_ValidY = train_test_split(train_BloodCell_x, train_BloodCell_y, train_size = 0.85)

#We dont make changes to validation set.
X_valid_final_Dermoscopic = Dermoscopy_ValidX
y_valid_final_Dermoscopic = Dermoscopy_ValidY
X_valid_final_BloodCell = BloodCell_ValidX
y_valid_final_BloodCell = BloodCell_ValidY

#Reshapes the values of the validations set to make them as we want to.
X_valid_final_Dermoscopic = X_valid_final_Dermoscopic.reshape(-1, 28, 28, 3)
X_valid_final_BloodCell = X_valid_final_BloodCell.reshape(-1, 28, 28 ,3)
   
#We make the labels being binary in order to pass them for smote_function.
y_valid_final_Dermoscopic = to_categorical(y_valid_final_Dermoscopic)
y_valid_final_BloodCell = to_categorical(y_valid_final_BloodCell)

#Final X and Y for dermoscopy data set.
x_train_reshaped_Dermoscopic, y_train_reshaped_Dermoscopic = smote_function(Dermoscopy_subsetX, Dermoscopy_subsetY)
#Final X and Y for BloodCell data set.
x_train_reshaped_BloodCell, y_train_reshaped_BloodCell = smote_function(BloodCell_subsetX, BloodCell_subsetY)                                                 

#In order to cocncatenate both datasets labels we make 3 more collumns in first dataset.
y_valid_final_Dermoscopic = np.concatenate((y_valid_final_Dermoscopic, np.zeros((y_valid_final_Dermoscopic.shape[0], 3))), axis = 1)
y_train_reshaped_Dermoscopic = np.concatenate((y_train_reshaped_Dermoscopic, np.zeros((y_train_reshaped_Dermoscopic.shape[0], 3))), axis = 1)

#Concatenate both datasets.
x_train_reshaped_All = np.concatenate((x_train_reshaped_Dermoscopic, x_train_reshaped_BloodCell), axis = 0)
y_train_reshaped_All = np.concatenate((y_train_reshaped_Dermoscopic, y_train_reshaped_BloodCell), axis = 0)
X_valid_final = np.concatenate((X_valid_final_Dermoscopic, X_valid_final_BloodCell), axis = 0)            
y_valid_final = np.concatenate((y_valid_final_Dermoscopic, y_valid_final_BloodCell), axis = 0)

                                                        #Train model#

#Iniate the neural newtwork.
CNN = network()
#regularization.
Callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights = True) # early stopping
#Our relatory.
history = CNN.fit(x_train_reshaped_All, y_train_reshaped_All, epochs = 200, validation_data = (X_valid_final, y_valid_final), callbacks = [Callback], class_weight = class_weights)

#Evaluating the model with Balanced accuracy score.
y_predict = np.argmax(CNN.predict(X_valid_final), axis=-1) 
y_true = np.argmax((y_valid_final), axis=-1)

#Calculate Balanced Accuracy.
BalAccuracy = metrics.balanced_accuracy_score(y_true, y_predict)

#If it's the first iteration, then best_BalAccuracy will be always 0.
if first_iteration:
    best_BalAccuracy = 0
    first_iteration = 0

#If new BallAccuracy is better than the bestAccuracy then we save the predict and the plot (if show_plots is activated).
if BalAccuracy > best_BalAccuracy:
                    
    #Assign to the best_BalAccuracy the new best BalAccuracy
    best_BalAccuracy = BalAccuracy
    
                        #Predict#
    
    X_test = X_test.reshape(len(X_test),28,28,3)
    Y_test = CNN.predict(X_test)
    Y_test = np.argmax(Y_test, axis=-1) 
        
    #Save the prediction
    np.save('Ytest_Classification2', Y_test) 

    if show_plots:
        
        #Plot and then save them as PNG files.
        Plot_and_Save()   
        
        