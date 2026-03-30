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
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

#Loads the data
def LoadData():
    
    # Loading data from the given files
    Xtrain = np.load('Xtrain_Classification1.npy')
    ytrain = np.load('ytrain_Classification1.npy')
    Xtest = np.load('Xtest_Classification1.npy')
    
    return Xtrain, ytrain, Xtest

#Just to verify the imbalance
def print_imbalance(y):
    
    num_nevu = len([0 for i in range(y.shape[0]) if y[i] == 0])
    num_melanoma = len([1 for i in range(y.shape[0]) if y[i] == 1])
    
    total = num_nevu + num_melanoma
    
    print(f"class 0 (nevu): {num_nevu} ({num_nevu / total * 100:.2f} %)")
    print(f"class 1 (melanoma): {num_melanoma} ({num_melanoma / total * 100:.2f} %)")
    
#Tests the various ways of achieving results.
def AnalyzeData(smote, oversample, equalize_data, subsampling, X_train_subset, y_train_subset, X_valid, y_valid):
    
    if smote:
        # smote_seed = random.randint(1, 1000)
        smote = SMOTE(sampling_strategy='auto', k_neighbors = 10)
        x_train_reshaped, y_train_reshaped = smote.fit_resample(X_train_subset, y_train_subset)

        x_train_reshaped = X_train_subset.reshape(-1,28,28,3)
        y_train_reshaped = to_categorical(y_train_subset)
    
    #We wont use but was tested
    elif subsampling:
        
        x_train_reshaped, y_train_reshaped = subsample(X_train_subset, y_train_subset)
        
        x_valid_reshaped = X_valid.reshape(-1,28,28,3)
        y_valid_reshaped = to_categorical(y_valid)
        
    #We wont use but was tested    
    elif equalize_data:
        
        #DATA AUGMENTATION for classes equalization
        x_train_reshaped, y_train_reshaped = equalizing(X_train_subset, y_train_subset)
       
        if oversample:
            
            samples_to_increase = 2000
            x_train_reshaped, y_train_reshaped = oversampling(x_train_reshaped, y_train_reshaped, samples_to_increase)
    
    #Just a reshaoe then
    else:
        
        #Reshapes for 28x28x3
        x_train_reshaped = X_train_subset.reshape(-1,28,28,3)
        x_valid_reshaped = X_valid.reshape(-1,28,28,3)

        y_train_reshaped = to_categorical(y_train_subset)
        y_valid_reshaped = to_categorical(y_valid)
    
    return x_train_reshaped, y_train_reshaped
    
#Return the values required
def countSamples(y):
    
    num_nevu = len([0 for i in range(y.shape[0]) if y[i] == 0])
    num_melanoma = len([1 for i in range(y.shape[0]) if y[i] == 1])
    total = num_nevu + num_melanoma
    
    return num_nevu, num_melanoma, total

#If we want to plot the images we will test in validation, we used this.
# def plot_dataset(x, y, classes):
    
#     subplot = int(np.ceil(np.sqrt(np.size(y))))
#     figure = subplot * 2
#     plt.figure(figsize=(figure, figure))

#     for i in np.array(range(np.size(y))):
#         plt.subplot(subplot, subplot, i + 1)
#         plt.xticks([])
#         plt.yticks([])
#         # plt.grid(False)
#         plt.imshow(
#             x[i, :].reshape(28, 28, 3),
#             origin = "lower",
#             alpha = 1,
#             aspect = "auto",
#             cmap = "viridis",
#             interpolation = "nearest",
#         )
#         # plt.xlabel(classes[int(y[i])])
#     plt.show()


# Calculate the balanced accuracy
def BalancedAccuracy(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    
    # Calculate the confusion matrix
    conf_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=2)
    
    #True Postive, True Negative, False Positive, False Negative values
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    
    C0 = TN + FP
    C1 = TP + FN
    
    #Just to ensure that we dont have Nans.
    if C0 == 0:
        specificity = 0.0  # Ensure specificity is of type float
    else:
        specificity = float(TN) / float(C0)
    
    #Just to ensure that we dont have Nans.
    if C1 == 0:
        sensitivity = 0.0  # Ensure sensitivity is of type float
    else:
        sensitivity = float(TP) / float(C1)
    
    BalancedAccuracy = (sensitivity + specificity) / 2.0
    
    return BalancedAccuracy, specificity, sensitivity

#Generates data
def dataGenerator(oversample, augmentation):
    
    Datagen = ImageDataGenerator(rotation_range = 90 if oversample == 1 else 50, 
        width_shift_range = 0.7,
        height_shift_range = 0.3,
        zoom_range = 0.5,
        brightness_range = [0.5, 1.5],
        horizontal_flip = augmentation == 1, #augmentation = 1, TRUE; #augmentation = 0, FALSE 
        vertical_flip = False,
        fill_mode = 'nearest')
    
    # Return/computes quantities required for featurewise normalization
    return Datagen

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
    
    #The final dense layer with 2 units and softmax activation is used for binary classification. 
    #This layer will produce probability scores for the two classes.
    model.add(tf.keras.layers.Dense(2, activation ='softmax'))
    #model.summary()

                                    #Compile#
                        
    #The model is compiled with the Adam optimizer, a binary cross-entropy loss function, and accuracy 
    #and Balanced Accuracy as evaluation metrics. The learning rate for the optimizer is set to 0.0002 (The smaller the better).                  
    model.compile(tf.keras.optimizers.Adam(0.0002),loss=tf.keras.losses.BinaryCrossentropy(), 
                  metrics=['accuracy', BalancedAccuracy])   #tf.keras.metrics.CategoricalAccuracy(name='accuracy')
    
    return model

#When the data is oversampled
def oversampling(x, y, additional_samples):
    
    GeneratesData = dataGenerator(1, 0)
    GeneratesData.fit(x)
    
    # Create more data
    y = np.argmax(y, axis=-1)
    DataAugmentation = GeneratesData.flow(x, y, batch_size=32)

    new_values = []
    new_labels = [] #This labels are refeering to new_values
    
    
    # Extract
    for j in range(len(DataAugmentation)):
        if j * 32 > additional_samples:
            break #We dont want to extract anymore.
        
        new_data, new_label = next(DataAugmentation[j] for i in range(32))
        new_values.append(new_data)
        new_labels.append(new_label)
    
    values = np.concatenate(new_values, axis = 0)
    labels = np.concatenate(new_labels, axis = 0)
    
    conc_values = np.concatenate((x, values), axis=0)
    conc_labels = np.concatenate((y, labels), axis=0)

    yfinal = np.concatenate((y, conc_labels))
    Xfinal = np.concatenate((x, conc_values))

    #shuffle
    mix = np.random.permutation(len(conc_labels))
    yfinal = yfinal[mix]
    Xfinal = Xfinal[mix]
    
    #One-Hot Encoding# 
    yfinal = to_categorical(yfinal)
    
    return Xfinal, yfinal

#When the data is subsampled
def subsample(x, y):
    
    labels = y
    
    #number of nevu , melanoma and the total count
    N_nevu = len([0 for i in range(labels.shape[0]) if labels[i] == 0]) #Sempre que y = 0 count++
    N_melanoma = len([1 for i in range(labels.shape[0]) if labels[i] == 1])
    
                        #Working on extra number of samples#
    N_nevu_to_remove = N_nevu - N_melanoma # What is extra is removed
    LastIndex_nevu = N_nevu - N_nevu_to_remove
    
                                #Working on Nevu class#
    New_nevu = np.array([x[i, :] for i in range(len(labels)) if labels[i] == 0 ])
    
                    #Working on Nevu class witouth the excedentarys#
    New_nevu = New_nevu[:LastIndex_nevu, :]
    N_new_nevu = New_nevu.shape[0]
    New_nevu_label = [0 for i in range(N_new_nevu)]

                             #Working on Melanoma class#
    melanoma = np.array([x[i, :] for i in range(len(labels)) if labels[i] == 1])
    melanoma = melanoma[:, :]
    melanoma_label = [1 for i in range(N_melanoma)]
    
    #Concatenate both classes
    conc_values = np.concatenate((np.array(melanoma), np.array(New_nevu)), axis = 0)
    conc_labels = np.concatenate((np.array(melanoma_label), np.array(New_nevu_label)))
    
    #shuffle
    mix = np.random.permutation(len(conc_labels))
    yfinal = conc_labels[mix]
    Xfinal = conc_values[mix].reshape(-1,28,28,3)
    
    #One-Hot Encoding# 
    yfinal = to_categorical(yfinal)
    
    return Xfinal, yfinal

#Data Augmentantion
def equalizing(x, y):
    
    #Start the array for both classes
    nevu = np.array([])
    melanoma = np.array([])
    
    Melanoma_Augmentation = np.array([])
    
    #y = 0
    nevu = np.array([x[i, :] for i in range(len(y)) if y[i]==0])
    #y = 1
    melanoma = np.array([x[i, :] for i in range(len(y)) if y[i]==1])
    
    N_nevu = int(nevu.shape[0])
    N_melanoma = int(melanoma.shape[0])
    Melanoma_reshaped = np.reshape(melanoma, (N_melanoma, 28, 28, 3))
    
    GeneratesData = dataGenerator(0 , 1) #It gives me GeneratesData.fit(Melanoma_reshaped) ?
    GeneratesData.fit(Melanoma_reshaped)
    
    # Now , we create more data 
    AugmentationData = GeneratesData.flow(Melanoma_reshaped, batch_size = 32)
    
    equal = N_nevu - N_melanoma
    new_data = []
    # A loop is used to extract and store newly augmented "melanoma" data from the data_aug generator. 
    #This is done until the desired number of additional samples (equal) is reached.
    
    for j in range(len(AugmentationData)):
        
        if j * 32 > equal:
            break
        
        new = next(AugmentationData[j] for i in range(32))
        new_data.append(new)
        
    Melanoma_Augmentation = np.concatenate(new_data, axis = 0)
    melanoma = melanoma.reshape(len(melanoma), 28, 28, 3)
    nevu = nevu.reshape(len(nevu), 28, 28, 3)
    
    #We now concatenate both.
    Melanoma_Augmentation = np.concatenate((Melanoma_Augmentation, melanoma), axis=0)
    N_melanoma_aug = Melanoma_Augmentation.shape[0]
    
                                    #Creating the labels#
                                    
    #All samples will be filled with number 0 as we want.
    Nevu_label = np.array([0 for i in range(N_nevu)])
    #All samples will be filled with number 1 as we want.
    MelanomaAugmentation_label = np.array([1 for i in range(N_melanoma_aug)])
    
    #Concatenating
    conc_values = np.concatenate((nevu, Melanoma_Augmentation)) 
    conc_labels = np.concatenate((Nevu_label, MelanomaAugmentation_label))
    
    #shuffle (The data is shuffled randomly to ensure that the order of the samples does not affect the model's training.)
    mix = np.random.permutation(len(conc_labels))
    ytrain_aug = conc_labels[mix]
    Xtrain_aug = conc_values[mix]

                                        #One-Hot Encoding# 
    #it's used to convert the class labels from the "nevus" and "melanoma" classification problem into a format suitable for training a machine learning model
    ytrain_aug = to_categorical(ytrain_aug)
    #it converts 0 to [1, 0] (nevus) and 1 to [0, 1] (melanoma).
    
    return Xtrain_aug, ytrain_aug
                                            #flags used#

#Testing the subsampling case ((0 - not used))
subsampling = 0

#Tested with data augmentation (0 - not used)
oversample = 0 

#The one we got better results (1 - used)
smote = 1

#Testing the data agumentation (0 - not used)
equalize_data = 0 

#For plotting the graphics in need (1 - They are plotted, 0 - not plotted)
plot_costs = 1 

#training with class weight 
classWeights = 1

X_train, y_train, X_test = LoadData()
#Print the imbalance for the dataset loaded.
print_imbalance(y_train)

#Normalization 8-bit, 
#the original pixel values in an 8-bit image typically range from 0 (black) to 255 (white)
X_train = X_train / 255 
X_test = X_test / 255
#Now the scale is between [0, 1].

if classWeights:
    
    nevu, melanoma, total_samples = countSamples(y_train)
    
    weight_nevu = (1 / nevu) * (total_samples / 2.0)
    weight_melanoma = (1 / melanoma) * (total_samples / 2.0)
    class_weights = {0: weight_nevu, 
                     1: weight_melanoma}

    print("Weight of class 0: {:.2f}".format(weight_nevu))
    print("Weight of class 1: {:.2f}".format(weight_melanoma))


#Get validation set (15% for validation set)
X_train_subset, X_valid, y_train_subset, y_valid = train_test_split(X_train, y_train, train_size=0.85)
#Print the imbalance for the new dataset
print_imbalance(y_train_subset)
#make a little subsample to have 50/50 valid set 
X_valid_final, y_valid_final = subsample(X_valid, y_valid)

x_train_reshaped, y_train_reshaped = AnalyzeData(smote, oversample, equalize_data, subsampling,
                                                 X_train_subset, y_train_subset, X_valid, y_valid)

                                           #Train model#

#Iniate the neural newtwork
CNN = network()

#regularization
Callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights = True) # early stopping     #class_weight = 

#Our relatory
history = CNN.fit(x_train_reshaped, y_train_reshaped, epochs = 200, validation_data = (X_valid_final, y_valid_final), callbacks = [Callback], class_weight = class_weights)

#Evaluating the model with Balanced accuracy score.
y_predict = np.argmax(CNN.predict(X_valid_final), axis=-1) 
y_true = np.argmax((y_valid_final), axis=-1)

BalAccuracy = metrics.balanced_accuracy_score(y_true, y_predict)
print('Bal-Accuracy-Score', ':',BalAccuracy)

# Plot cost and accuracy plots 
if plot_costs:

    #Customize the font size for labels
    label_fontsize = 12
    legend_fontsize = 6
    tick_label_fontsize = 8
    
    # Set the figure size
    plt.figure(figsize=(12, 3))  
    
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
    plt.plot(
        history.history["BalancedAccuracy"],
        label="training set", color="orange"
    )
    plt.plot(
        history.history["val_BalancedAccuracy"],
        label="validation set", color="red"
    )
    plt.legend(fontsize=legend_fontsize)
    
    # Adjust tick label font size
    for ax in plt.gcf().get_axes():
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(tick_label_fontsize)
    
    # Adjust subplot layout to prevent overlapping labels
    plt.subplots_adjust(wspace=0.5)  # Adjust the horizontal space between subplots
    
    # Show the plots.
    plt.show()

                                     #Predict#

X_test = X_test.reshape(len(X_test),28,28,3)
Y_test = CNN.predict(X_test)
Y_test = np.argmax(Y_test, axis=-1) 
 
#Save the prediction
np.save('Ytest_Classification1', Y_test) 