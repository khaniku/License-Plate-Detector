import os
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from skimage.io import imread
from skimage.filters import threshold_otsu

if __name__ == '__main__':
    characters = [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
                'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z'
            ]

    def read_data(dir):
        image_data = []
        target_data = []
        for char in characters:
            for each in range(10):
                image_path = os.path.join(dir, char, char + '_' + str(each) + '.jpg')
                img_details = imread(image_path, as_gray=True)
                # converts each character image to binary image
                binary_image = img_details < threshold_otsu(img_details)
                flat_bin_image = binary_image.reshape(-1)
                image_data.append(flat_bin_image)
                target_data.append(char)

        return (np.array(image_data), np.array(target_data))

    def cross_validation(model, num_of_fold, train_data, train_label):
        # this uses the concept of cross validation to measure the accuracy
        # of a model, the num_of_fold determines the type of validation
        accuracy_result = cross_val_score(model, train_data, train_label,
                                          cv=num_of_fold)
        print("Cross Validation Result for ", str(num_of_fold), " -fold")

        print(accuracy_result * 100)

    print('reading data')
    training_dataset_dir = './data'
    image_data, target_data = read_data(training_dataset_dir)
    print('reading data completed')

    svc_model = SVC(kernel='linear', probability=True)

    cross_validation(svc_model, 4, image_data, target_data)

    print('training model')

    svc_model.fit(image_data, target_data)

    print("Training completed")
    filename = './model.sav'
    pickle.dump(svc_model, open(filename, 'wb'))
    print("Model saved!")