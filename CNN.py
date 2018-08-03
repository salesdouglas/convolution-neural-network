from keras.datasets import fashion_mnist

(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import keras

from keras.models import Sequential,Input,Model
from keras.layers import Dense,Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU



print ('Training data shape: ', train_X.shape, train_Y.shape)

print ('Testing data shaoe: ', test_X.shape, test_Y.shape)

#From the abouve output, you can see that the training data has a shape of 60000 x 28 x 28 since there are 600000 training samples 
#each of 28 x 28 dimension. Similiarly, the test data has a shape of 10000 testing samples.

#Find the unique numbers from the train leables 
classes = np.unique(train_Y)

nClasses = len(classes)

print ('Total number of outputs : ', nClasses)
print ('Output classes : ', classes)

#There's also a total of ten output classes that range from 0 to 9

#Also, dont forget to take a look at what the images in your datasets

plt.figure(figsize=[5,5])


#Display the first image in training data 

plt.subplot(121)

plt.imshow(train_X[0,:,:], cmap='gray')

plt.title("Ground Truth: {}" .format(train_Y[0]))


plt.subplot(121)

plt.imshow(test_X[0,:,:], cmap='gray')

plt.title("Ground Truth: {}" .format(test_Y[0]))

#As a first step, convert each 28 x 28 image of the train and test set into a matrix of size 28 x 28 x 1 which is fed into the network

train_X = train_X.reshape(-1,28,28,1)

test_X = test_X.reshape(-1,28,28,1)

train_X.shape, test_X.shape

#The data right now is in an int8 format, so before you fedd it into the network you need to convert it is type to float32, and you also hae to rescal the pixel values in range 0 -1 inclusive. so let's do that


train_X = train_X.astype('float32')


test_X = test_X.astype('float32')


train_X = train_X/255. 

test_X = test_X/255.

#Now you nedd to convert the class labels into a one-hota enconding vector

#In one-hot enconding, you convert the categorical data into a vector of numbers. 
#The reason why you convert the categorical data in one hot enconding is that machine learning algorithms can't work with categorical data directly.
#You generate one boolean column for each category or class. Only one of these coluns could take on the value 1 for each sample.

#For your problema statement, the one hot encoding will be a row vector, and for each image, it ill have a dimension of 
#1x10. The important thing to note here is that the vector consists of all zeros except for the class that it represents,
#and for that, it is 1. For example, the ankle boot image that you plotted above has a label of 9, so for all the ankle boot images
#the one hot encoding vector would be [0 0 0 0 0 0 0 0 1 0].

#So let's convert the training and testing labels into one-hot enconding vectors:

# Change the labels from caterical to one-hot encoding

train_Y_one_hot = to_categorical(train_Y)

test_Y_one_hot = to_categorical(test_Y)

#display the change for cateory label using one-hot encondigw


print ('Original label:' , train_Y[0])
print('After conversion to one-hot', train_Y_one_hot[0])

#This last step is a crucial one. In machine learning or any data specific task, you should partition the data correctly.
#For the model to generalize well, you split the training data into two parts, one designed for training and another one for validation. 
#In this case, you will train the model on 80% of the training data and validade it on 20% of the remaining training data.
#This will also help to reduce overfitting sice you will be validating the model on the data it would not have seen in training phase

train_X,valid_X,train_label,valid_label =  train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

#For one last time lets check  the shape of training and validation set.


#You wiil use a batch size of 64 using a higher batch size of 128 or 256 is also preferable it all depends on the memory . It contributes 
#massively to determining the learning parameters and affects the prediction accuracy. You will train the network for 20 epochs. 


batch_size = 64

epochs = 20

num_classes = 10

#fashion_model = Sequential()

#fashion_model.add(Conv2D(32, kernel_size=(3,3), activation='linear',input_shape=(28,28,1),padding='same'))

#fashion_model.add(LeakyReLU(alpha=0.1))

#fashion_model.add(MaxPooling2D((2,2), padding='same'))

#fashion_model.add(Conv2D(64, kernel_size=(3,3), activation='linear',padding='same'))

#fashion_model.add(LeakyReLU(alpha=0.1))

#fashion_model.add(MaxPooling2D((2,2), padding='same'))

#fashion_model.add(Flatten())

#fashion_model.add(Dense(128, activation='linear'))

#fashion_model.add(LeakyReLU(alpha=0.1))

#fashion_model.add(Dense(num_classes, activation='softmax'))

#After the model is created, you compile it using the Adam optimizer, same one of the most popular optimization algorithms.
#Additionally, you specify the loss type which is categorical cross entropy which is used for multi-class classifcation, you can
#also use binary cross- entropy as the loss function. Lastly, youspecify the metrics as accuracy which you wan5 to analyzer while the model is training

#fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

#Let's visualize the layers that you created in the above step by using the summary function. This will show
#some parameters(weights and biases) in each layer and also the total parameters in your model.

#fashion_model.summary()

#It's finally time to train the model with Keras fit funtion. The model trains for 20 epochs. The fit(0 function will return a 
#history object. By storying the result of this function in fashion_train, you can use it later to plot the accracy and loss function plots
#between  training andvalidation which will help you to analyze your model's performance visually

#fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

#test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)

#print ('Test loss: ', test_eval[0])

#print ('Test accuracy: ', test_eval[1])

#accuracy = fashion_train.history['acc']

#val_accuracy = fashion_train.history['val_acc']

#loss = fashion_train.history['loss']

#epochs= range(len(accuracy))

#plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

#plt.legend()

#plt.figure()

#plt.plot(epochs, loss, 'bo', label='Training loss')

#plt.plot(epochs, loss, 'b', label='Validation accuracy')

#plt.title ('Training and validation loss')

#plt.legend()

#plt.show()

#So let's create, compile and train the network again but this time with dopour. And run it for 20 epochs with a batch size of 64


fashion_model = Sequential()

fashion_model.add(Conv2D(32, kernel_size=(3,3), activation='linear',padding='same', input_shape=(28,28,1)))

fashion_model.add(LeakyReLU(alpha=0.1))

fashion_model.add(MaxPooling2D((2,2), padding='same'))

fashion_model.add(Dropout(0.25))

fashion_model.add(Conv2D(64, kernel_size=(3,3), activation='linear',padding='same'))

fashion_model.add(LeakyReLU(alpha=0.1))

fashion_model.add(MaxPooling2D((2,2), padding='same'))

fashion_model.add(Dropout(0.25))

fashion_model.add(Conv2D(128,(3,3), activation= 'linear', padding='same'))

fashion_model.add(LeakyReLU(alpha=0.1))

fashion_model.add(Flatten())

fashion_model.add(Dense(128, activation='linear'))

fashion_model.add(Dropout(0.3))

fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.summary()

fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

fashion_train_dropout = fashion_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(valid_X,valid_label))


#Let's save the model so that you can directly load  it and not have to train it again for  20 epochs. This way, you can load the model later on if you 
#need it and modify the architecture. Alternatively, you can start the training process on this saved model. It is always a good idea to save the model
#and even the model's weights .Because it saves you time.  Note that you can issue also save the model after every epoch so that, if some issue occurs 
#that stops the training at an epoch, you will ot have to start the training from the begginning

fashion_model.save("fashion_model_dropout.h5py")

#Model evaluation on the test set

#Finally, let's also evaluate your new model and see how it performs

test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=1)

print ('Test loss: ', test_eval[0])

print ('Test accuracy: ', test_eval[1])

accuracy = fashion_train_dropout.history['acc']

val_accuracy = fashion_train_dropout.history['val_acc']

loss = fashion_train_dropout.history['loss']

val_loss = fashion_train_dropout.history['val_loss']

epochs= range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'b' , label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, loss, 'b', label='Validation accuracy')

plt.title ('Training and validation loss')

plt.legend()

plt.show()

#Finally, you can see that the validation loss and validation accuracy both are in sync with the training loss and training accuracy. Ever though
#the validation loss and accuracy line are not linear, but it shows that your model is not overfitting: the validation loss is decreasing and not increasing
#and there is not much gap between training and validation accuracy

#Therefore,you can say that your model's generalization capability became much better since the loss on both test set and validation
#set was only slightly more compared to the training loss

predicted_classes = fashion_model.predict(test_X)

#Since the predictions yout get are floating point values, it will not be feasible to compare the predictted lable swith true test labels
#SO, you will round off the output which will convert the float values into an integer. Furher, you will use np.argmax() to select the index number 
#which has a highher value in a row

#For example, let's assume a prediction for one test imagem to be 0 1 0 0 0 0 0 0 0 0, the output for this should be a class label 1.

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

correct = np.where(predicted_classes==test_Y)[0]

print ("Found %d correct labels" % len(correct))

for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted{}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()
    
incorrect = np.where(predicted_classes!=test_Y[0])
print ("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class{}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()
    
    





