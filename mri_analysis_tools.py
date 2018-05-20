#!/usr/local/bin/python3
import gc
import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
import pyrenn as prn
from matplotlib.image import imsave
from skimage.filters import gaussian
from skimage.exposure import adjust_gamma
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


class ImageShower():
    """ class for showing a scrollable 3d image """
    def __init__(self, ax, image):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.image = image
        _, _, self.slices = image.shape
        self.ind = self.slices // 2

        self.show_instance = ax.imshow(self.image[:, :, self.ind], cmap='gray')
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.show_instance.set_data(self.image[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.show_instance.axes.figure.canvas.draw()


class MRI:
    ''' class represents MRI image '''
    def __init__(self, path_to_mri):
        try:
            self.image = nib.load(path_to_mri).get_fdata()
            self.path = path_to_mri
            self.shape = self.image.shape
        except:
            print('incorrect path: %s\n failed to load' % path_to_mri)

    def get_slice(self, slice):
        return self.image[:, :, slice]

    def show(self):
        """ shows the 3d (scrollable) mri image """
        fig, ax = plt.subplots(1, 1)
        image = ImageShower(ax, self.image)
        fig.canvas.mpl_connect('scroll_event', image.onscroll)
        plt.show()


class Data:
    """ class represents the following set of MRI images of a patient:
        t1, t2, t1ce, flair, tumor segmentation """
    def __init__(self, dir_name, **types):
        """ types should be presented as {mri_type : path} dictionary """
        self.dir = dir_name
        self.images = {}
        self.mri_types = ['t1', 't2', 't1ce', 'flair', 'seg']
        for mri_type, path in types.items():
            self.add(mri_type, path)

    def add(self, mri_type, path):
        if mri_type not in self.mri_types:
            print('incorrect type: %s' % mri_type)
            print('only following types are suitable: %s' % str(self.mri_types))
            return
        self.images.update({mri_type: MRI(path)})

    def show(self, mri_type):
        """ shows the 3d (scrollable) mri image """
        self.images.get(mri_type).show()


class DataStorage:
    """ class represents the collection of MRI images
        the unique key of an Data element is a name of a folder containing images """
    def __init__(self, **data):
        """ data should be like {dir_name : [mri_types]} """
        self.data = []
        for dir_name, mri_types in data.items():
            self.add(dir_name, *mri_types)
            print("%s added" % dir_name)

    def add(self, dir_name, *types):
        """ types should be in ['t1', 't2', 't1ce', 'flair', 'seg'] """
        for d in self.data:
            if d.dir == dir_name:
                print('Data contained in %s is already storaged.\nRemove it first to update the data.' % dir_name)
                return
        mri_types = {}
        for mri_type in types:
            mri_types.update({mri_type : dir_name + '_' + mri_type + '.nii.gz'})
        # .nii.gz is a necessary thing to operate the avaliable data
        # nibabel lib is here for working with nii format
        self.data.append(Data(dir_name, **mri_types))

    def remove(self, dir_name):
        """ removes Data which unique dir is dir_name """
        # todo: find out how to free memory after deletion
        if self.is_belong(dir_name):
            for d in self.data:
                if d.dir == dir_name:
                    del(d.images)
                    self.data.remove(d)
                    return

    def is_belong(self, dir_name):
        """ returns True if there are images in specified directory
            returns False otherwise """
        for d in self.data:
            if d.dir == dir_name:
                return True
        return False

    def get_slices(self, slice, *types):
        """ returns dictionary {type : [slices]} for specific slice
            types should be in ['t1', 't2', 't1ce', 'flair', 'seg'] """
        data = {}
        for t in types:
            slices = list(map(lambda x: x.images.get(t).get_slice(slice), self.data))
            data.update({t : slices})
        return data


class DataProcessing:
    """ a class for processing 2d mri images of same type """
    def __init__(self, data, gauss_sigma=0.5, gamma=1.5, auto_process=False):
        """ data is a list of 2d images
            auto_process runs all the processing functions with default values """
        self.images = data
        self.mean = self._common_mean()
        self.raw_images = [self._image_to_list(d) for d in self.images]
        self.gauss_sigma = gauss_sigma
        self.gamma = gamma
        if auto_process:
            self.apply_gaussian_filter()
            self.apply_gamma_correction()
            self.apply_normalization()
            self.apply_outliers_correction()
            self.apply_intensity_normalization()

    def _image_to_list(self, image):
        data = []
        for x in image:
            data += list(x)
        data.sort()
        return data

    def _common_mean(self):
        mean = 0
        for image in self.images:
            mean += image.mean()
        mean /= len(self.images)
        return mean

    def _outliers_correction(self, matrix, top_value):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > top_value:
                    matrix[i, j] = top_value

    def _extract_features(self, matrix, max_value=64):
        ''' matrix is a grayscalse picture,
            max_value is a value of a maximum grayscale value in a picture,
            output is a vector consists of the next features:
            contrast, dissimilarity, homogeneity, ASM, energy, correlation
        '''
        matrix = matrix.astype('int8') # GLCM requires integer values
        common_features = [0] * 6
        features = []
        feature_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        glcm1 = greycomatrix(matrix, [0], [1], max_value + 1, symmetric=True, normed=True)
        glcm2 = greycomatrix(matrix, [1], [1], max_value + 1, symmetric=True, normed=True)
        glcm3 = greycomatrix(matrix, [1], [0], max_value + 1, symmetric=True, normed=True)
        glcm4 = greycomatrix(matrix, [-1], [-1], max_value + 1, symmetric=True, normed=True)
        glcms = [glcm1, glcm2, glcm3, glcm4]
        for glcm in glcms:
            for i in range(len(feature_names)):
                feature = greycoprops(glcm, feature_names[i])[0][0]
                features += [feature]
                common_features[i] += feature
        features += [0] * 6
        for i in range(len(common_features)):
            features[-6 + i] = common_features[i] / len(glcms)
        print(len(features))
        return features

    def apply_gaussian_filter(self):
        for image in self.images:
            image = gaussian(image, sigma=self.gauss_sigma)

    def apply_gamma_correction(self):
        for image in self.images:
            image = adjust_gamma(image, gamma=self.gamma)

    def apply_normalization(self):
        for image in self.images:
            image = (image - image.mean()) / image.std()
            image *= self.mean / image.mean()

    def apply_outliers_correction(self, percent=2):
        number_of_images = len(self.images)
        for i in range(number_of_images):
            top_value = self.raw_images[i][-number_of_images * percent // 100]
            print(top_value)
            self._outliers_correction(self.images[i], top_value)

    def apply_intensity_normalization(self, max_value=64):
        for i in range(len(self.images)):
            self.images[i] *= max_value / self.images[i].max()

    def get_features(self):
        return list(map(lambda x: self._extract_features(x), self.images))


class Classifier:
    """ a class for classification input data using ann, svm, k-nn """
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None):
        """ add data to run classifiers with no input data """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
    def _check_result(self, res, y):
        """ private function for calculation accuracy, precision and recall """
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for i in range(len(res)):
            if type(y[0]) == list:
                ans = y[i][0]
            else:
                ans = y[i]
            if ans == 0:
                if res[i] == 0:
                    tp += 1
                else:
                    fp += 1
            elif ans == 1:
                if res[i] == 0:
                    fn += 1
                else:
                    tn += 1
        if len(res) == 0:
            print('empty data')
            return
        accuracy = (tp + tn) / len(res)
        if tp + fn == 0:
            print('tp+fn=0')
            recall = -1
        else:
            recall = tp / (tp + fn)
        if tp + fp == 0:
            print('tp+fp=0')
            precision = -1
        else:
            precision = tp / (tp + fp)
        if tp + tn + fp + fn != len(res):
            print('error')
        return accuracy, precision, recall

    def train_nn(self, layers=[5, 10, 5], x_train=None, y_train=None, k_max=100, E_stop=1e-7):
        """ layers should be a list of integers like [5, 10, 5];
            datasets should be in shape like M x N,
            where N is a number of features and M is a number of instanses;
            k_max is a maximum number of iterations """
        if x_train is None:
            x_train = self.x_train
        if y_train is None:
            y_train = self.y_train

        x_train = np.array(x_train).transpose()
        y = []
        if type(y_train[0]) == list:
            y_train = list(map(lambda x: x[0], y_train))
        for i in y_train:
            if i == 0:
                y.append([1, 0])
            else:
                y.append([0, 1])
        y = np.array(y).transpose()

        amount_of_features = x_train.shape[0]
        self.nn = prn.CreateNN([amount_of_features] + layers + [2])
        self.nn = prn.train_LM(x_train, y, self.nn, k_max=k_max, E_stop=E_stop, verbose=True)    

    def test_nn(self, x_test=None, y_test=None):
        """ add x_test and y_test if you want to test a model using another data
            returns: accuracy, precision, recall """
        if x_test is None:
            x_test = self.x_test
        if y_test is None:
            y_test = self.y_test

        x_test = np.array(x_test).transpose()
        res = prn.NNOut(x_test, self.nn)

        if res.shape[1] != len(y_test):
            print('Incorrect length of y_test\nexpected: {}, got: {}'.format(res.shape[1], len(y_test)))
        result = []
        for i in range(res.shape[1]):
            result.append(np.argmax(res[:, i]))
        print(self._check_result(result, y_test))

    def train_svm(self, x_train=None, y_train=None):
        if x_train is None:
            x_train = self.x_train
        if y_train is None:
            y_train = self.y_train
        self.clf = svm.SVC(degree=9, probability=True, tol=0.0001)
        self.clf.fit(x_train, y_train)

    def test_svm(self, x_test=None, y_test=None):
        """ add x_test and y_test if you want to test a model using another data
            returns: accuracy, precision, recall """
        if x_test is None:
            x_test = self.x_test
        if y_test is None:
            y_test = self.y_test
        print(self._check_result(self.clf.predict(x_test), y_test))

    def train_knn(self, n_neib=3, x_train=None, y_train=None):
        """ n_neib is amount of neighbours """
        if x_train is None:
            x_train = self.x_train
        if y_train is None:
            y_train = self.y_train
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(x_train, y_train)

    def test_knn(self, x_test=None, y_test=None):
        """ add x_test and y_test if you want to test a model using another data
            returns: accuracy, precision, recall """
        if x_test is None:
            x_test = self.x_test
        if y_test is None:
            y_test = self.y_test
        print(self._check_result(self.knn.predict(x_test), y_test))

def show_processed_pics(pic, i):
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(pic, cmap='gray')
    ax = fig.add_subplot(2, 2, 2)
    temp = gf(pic, 0.5)
    temp = mat_dif(temp,  get_min(temp))
    ax.imshow(exposure.adjust_gamma(temp, i), cmap='gray')
    ax.imshow(gf(temp, i), cmap='gray')
    plt.show()

def amount_of_dots(matrix, t):
    amount = 0
    for i in range(240):
        for j in range(240):
            if matrix[i, j] == t:
                amount += 1
    return amount

def get_data(path, i):
    return nib.load(path).get_fdata()[:, :, i]

def get_answer(num, coordinates):
    return len(list(filter(lambda x: x[0] > num or x[1] < num, coordinates)))

def get_z_coordinates(name):
    data = nib.load(name).get_fdata()
    begin = 0
    end = 155
    for i in range(155):
        if empty_matrix_check(data[:, :, i]):
            begin = i
            break
    for i in range(154, -1, -1):
        if empty_matrix_check(data[:, :, i]):
            end = i
            break
    return [begin, end]

if __name__ == '__main__':
    pass
    


