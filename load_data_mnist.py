import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import random
import matplotlib.pyplot as plt
import pandas as pd

#
# MNIST Data Loader Class
#

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def _read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        
        images = np.array(image_data).reshape(size, rows * cols)
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self._read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self._read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        
        pixel_cols = [f"pixel_{i}" for i in range(784)]
        value_cols = [f"number_{i}" for i in range(10)]
   
        df_x_train = pd.DataFrame(x_train, columns=pixel_cols)
        df_x_test = pd.DataFrame(x_test, columns=pixel_cols)
        
        df_x_train = df_x_train / 255.0
        df_x_test = df_x_test / 255.0
        
        y_train_arr = np.array(y_train)
        y_test_arr = np.array(y_test)
        
        df_y_train = pd.DataFrame(np.eye(10, dtype=int)[y_train_arr], columns=value_cols)
        df_y_test = pd.DataFrame(np.eye(10, dtype=int)[y_test_arr], columns=value_cols)
        
        return (df_x_train, df_y_train),(df_x_test, df_y_test)        


def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1
    plt.show()

def load_mnist(print_img: bool = False) -> tuple[pd.DataFrame]:
    """Returns X_train, y_train, X_test and y_test as Dataframes"""
    
    input_path = './data'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    
    images_2_show = []
    titles_2_show = []
    
    for i in range(0, 10):
        r = random.randint(1, 59999) # Max is 59999 for 0-indexed train set
        images_2_show.append(x_train.iloc[r].values.reshape(28, 28)) 
        titles_2_show.append(str(y_train.iloc[r].argmax()))

    for i in range(0, 5):
        r = random.randint(1, 9999) # Max is 9999 for 0-indexed test set
        images_2_show.append(x_test.iloc[r].values.reshape(28, 28))
        titles_2_show.append(str(y_test.iloc[r].argmax()))    

    if print_img: show_images(images_2_show, titles_2_show)
    
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":   
    load_mnist()