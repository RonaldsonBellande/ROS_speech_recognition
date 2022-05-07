from header_imports import *


class model_training(model_building):
    def __init__(self, model_type, data_type):
        super().__init__(model_type, data_type)
        
        self.number_images_to_plot = 25
        self.batch_size = [10, 20, 40, 60, 80, 100]
        self.epochs = [1, 5, 15, 50, 100, 200]
        self.graph_path = "graph_charts/"
        self.model_path = "models/" + self.folder 
        self.param_grid = dict(batch_size=self.batch_size, epochs=self.epochs)
        self.callback_1 = TensorBoard(log_dir="logs/{}-{}".format(self.create_model_type, int(time.time())))
        self.callback_2 = ModelCheckpoint(filepath=self.model_path, save_weights_only=True, verbose=1)
        self.callback_3 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor= 0.5, min_lr=0.00001)
        
        self.train_model()
        self.evaluate_model()
        self.plot_model()
        self.plot_random_examples()
        
    

    def train_model(self):
       
        grid = GridSearchCV(estimator=self.model, param_grid=self.param_grid, n_jobs=1, cv=3, verbose=10)
        self.get_training_time("starting --: ")
        self.speech_model = self.model.fit(self.X_train, self.Y_train_vec,
                batch_size=self.batch_size[2],
                validation_split=0.10,
                epochs=self.epochs[1],
                callbacks=[self.callback_1, self.callback_2, self.callback_3],
                shuffle=True)

        self.get_training_time("ending --: ")
        self.model.save(self.model_path + self.create_model_type + "_speech_categories_"+ str(self.number_classes)+"_model.h5")
   

    def evaluate_model(self):
        evaluation = self.model.evaluate(self.X_test, self.Y_test_vec, verbose=1)

        with open(self.graph_path + self.folder + self.create_model_type + "_evaluate_speech_category_" + str(self.number_classes) + ".txt", 'w') as write:
            write.writelines("Loss: " + str(evaluation[0]) + "\n")
            write.writelines("Accuracy: " + str(evaluation[1]))
        
        print("Loss:", evaluation[0])
        print("Accuracy: ", evaluation[1])



    def plot_model(self):

        plt.plot(self.speech_model.history['accuracy'])
        plt.plot(self.speech_model.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig(self.graph_path + self.folder + self.create_model_type + '_accuracy_' + str(self.number_classes) + '.png', dpi =500)
        plt.clf()

        plt.plot(self.speech_model.history['loss'])
        plt.plot(self.speech_model.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig(self.graph_path + self.folder + self.create_model_type + '_lost_' + str(self.number_classes) +'.png', dpi =500)
        plt.clf()


    def plot_random_examples(self):

        plt.figure( dpi=256)
        predicted_classes = self.model.predict(self.X_test)
        
        for i in range(self.number_images_to_plot):
            plt.subplot(5,5,i+1)
            plt.axis('off')
            plt.title("Predicted - {}".format(self.category_names[np.argmax(predicted_classes[i], axis=0)]) + "\n Actual - {}".format(self.category_names[np.argmax(self.Y_test_vec[i,0])]),fontsize=5)
            plt.tight_layout()
            plt.savefig(self.graph_path + self.folder + self.create_model_type + '_prediction' + str(self.number_classes) + '.png', dpi =500)


    def get_training_time(self, start):

        date_and_time = datetime.datetime.now()
        test_date_and_time = "/test_on_date_" + str(date_and_time.month) + "_" + str(date_and_time.day) + "_" + str(date_and_time.year) + "_time_at_" + date_and_time.strftime("%H:%M:%S")

        with open(self.graph_path + self.folder + self.create_model_type + "_evaluate_training_time_" + str(self.number_classes) + ".txt", 'a') as write:
            write.writelines(start + test_date_and_time + "\n")




