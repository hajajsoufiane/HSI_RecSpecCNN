model_L3D = Sequential(name = 'RecSpecCNN')

model_L3D.add(Conv3D(16, kernel_size=(3, 3, 3),activation='linear',padding='same',input_shape=(3,3,ip_shape_3D[2],1)))
model_L3D.add(LeakyReLU(alpha=0.1))


model_L3D.add(Conv3D(32, (3, 3, 3), activation='linear',padding='same'))
model_L3D.add(LeakyReLU(alpha=0.1))


model_L3D.add(Conv3D(64, (3, 3, 3), activation='linear',padding='same'))
model_L3D.add(LeakyReLU(alpha=0.1))

model_L3D.add(MaxPooling3D(pool_size=(2, 2, 2),padding='same'))


model_L3D.add(Reshape((2,2*15*64)))  

model_L3D.add(Bidirectional(LSTM(units=64, return_sequences=True)))


model_L3D.add(GlobalAveragePooling1D()) 

model_L3D.add(Dense(64, activation='relu'))

model_L3D.add(Dropout(0.4))

model_L3D.add(Dense(n_outputs_3D, activation='softmax'))
model_L3D.summary()

model_L3D.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor = 'val_loss',
                            mode = 'min',
                            min_delta = 0,
                            patience = 10,
                            restore_best_weights = True)

checkpoint = ModelCheckpoint(filepath = 'RecSpecCNN.hdf5',
                             monitor = 'val_loss',
                             mode ='min',
                             save_best_only = True)

tensorboard = TensorBoard(log_dir='SA_logs/{}'.format(time()))

histL3Dep30 = model_L3D.fit(X_train_3D,
                       y_train_3D,
                       epochs = 100,
                       batch_size = 64,
                       validation_data = (X_val_3D, y_val_3D),
                       callbacks=[early_stop,checkpoint,tensorboard])

model_L3D.save('RecSpecCNN.hdf5')