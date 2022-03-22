from header_imports import *

class speech_building(object):
    def __init__(self, model_type):

        self.label_name = []
        self.number_classes = 3
        self.path  = "voice_data/"
        self.channel = 1
        self.number_mfcc = 2121
        self.mfcc_vectors = np.empty([self.number_mfcc, 128, 44]) 


        self.valid_sound = [".wav"]
        self.model = None

        self.model_summary = "model_summary/"
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.create_model_type = model_type
        self.categories = ["bed", "happy", "cat"]
        
        self.model_categories = self.categories
        self.data_to_array_label_sound()
        self.splitting_data_normalize()

        if self.create_model_type == "model1":
            self.create_models_1()
        elif self.create_model_type == "model2":
            self.create_models_2()
        elif self.create_model_type == "model3":
            self.create_model_3()
        elif self.create_model_type == "model4":
            self.create_models_4()
        elif self.create_model_type == "model5":
            self.create_model_5()

        self.save_model_summary()
        print("finished")


    def data_to_array_label_sound(self):

        for label in self.categories:
            self.wav_files = [self.path + label + '/' + i for i in os.listdir(self.path + '/' + label)]

            for wavfile in self.wav_files:
                wave, sr = librosa.load(wavfile)
                mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=self.number_mfcc)
                np.append(self.mfcc_vectors, mfcc)

                if label == "bed":
                    self.label_name.append(0)
                elif label == "happy":
                    self.label_name.append(1)
                elif label == "cat":
                    self.label_name.append(2)
        
        
        if self.create_model_type == "model4":
            self.mfcc_vectors =  self.mfcc_vectors.reshape(self.mfcc_vectors.shape[0], self.mfcc_vectors.shape[1], self.mfcc_vectors.shape[2])
        else:
            self.mfcc_vectors =  self.mfcc_vectors.reshape(self.mfcc_vectors.shape[0], self.mfcc_vectors.shape[1], self.mfcc_vectors.shape[2], 1)

        self.label_name = np.array(self.label_name)
        self.label_name = tf.keras.utils.to_categorical(self.label_name , num_classes=self.number_classes)



    def splitting_data_normalize(self):
        
        self.X_train, self.X_test, self.Y_train_vec, self.Y_test_vec = train_test_split(self.mfcc_vectors, self.label_name, test_size = 0.1, random_state=42)
        self.input_shape = self.X_train.shape[1:]
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train_vec, self.number_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test_vec, self.number_classes)
    

    def create_models_1(self):
        
        self.model = Sequential()

        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape = self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.number_classes, activation='softmax'))

        self.model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics= ['accuracy'])

        return self.model


    def create_models_2(self):

        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape = self.input_shape))
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units = self.number_classes, activation="softmax"))
        self.model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics= ['accuracy'])
	
        return self.model


    def create_model_3(self):

        self.model = Sequential()
        
        self.MyConv(first = True)
        self.MyConv()
        self.MyConv()
        self.MyConv()

        self.model.add(Flatten())
        self.model.add(Dense(units = self.number_classes, activation = "softmax", input_dim=2))
        self.model.compile(loss = "binary_crossentropy", optimizer ="adam", metrics= ["accuracy"])
        
        return self.model

        

    def MyConv(self, first = False):

        if first == False:
            self.model.add(Conv2D(filters=64, kernel_size=(4, 4),strides = (1,1), padding="same", input_shape = self.input_shape, activation="relu"))
        else:
            self.model.add(Conv2D(64,(4, 4),strides = (1,1), padding="same", input_shape = self.input_shape, activation="relu"))
    
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(filters=32, kernel_size = (4, 4),strides = (1,1),padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.25))

        

    def create_models_4(self):
        
        self.model = Sequential()
        self.model.add(LSTM(16, input_shape = self.input_shape, activation="sigmoid"))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.add(Dense(self.number_classes, activation='softmax'))
        self.model.compile(loss = "binary_crossentropy", optimizer ="adam", metrics= ["accuracy"])

        return self.model

    
    def create_model_5(self):
        
        self.model = Sequential()
        self.model.add(Dense(32,activation='sigmoid', input_shape = (42, 11)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3,activation='softmax'))
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

        return self.model



    def save_model_summary(self):
        with open(self.model_summary + self.create_model_type +"_summary_architecture_" + str(self.number_classes) +".txt", "w+") as model:
            with redirect_stdout(model):
                self.model.summary()


    



    
