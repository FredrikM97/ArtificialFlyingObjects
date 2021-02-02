# Default model:
    inputs = Input(shape=self.inputShape) 
    down = ConvLSTM2D(filters=5, kernel_size=(1, 1),
                   padding='same', return_sequences=True)(inputs) 

    final = Conv3D(filters=3, kernel_size=(1, 1, 3),
           activation='sigmoid', padding='same', data_format='channels_last')(down)

    model = Model(inputs=inputs, outputs=final)
    model.summary()untitled