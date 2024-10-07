from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator


class DataLoader:
    def __init__(self, train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size

    def get_data_generators(self):
        # Data augmentation for the training set
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Rescaling for validation and test sets (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1./255)

        # Create the training, validation, and test data generators
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

        val_generator = val_test_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

        test_generator = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

        return train_generator, val_generator, test_generator
