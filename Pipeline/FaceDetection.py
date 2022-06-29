import numpy as np
import math
import pickle
import cv2

def integral_image(image):
    integralImage = np.zeros(image.shape)
    row = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            if y-1 >= 0:
                row[y][x] = row[y-1][x] + image[y][x]
            else:
                row[y][x] = image[y][x]
            if x-1 >= 0:
                integralImage[y][x] = integralImage[y][x-1] + row[y][x]
            else: 
                integralImage[y][x] = row[y][x]
    return integralImage

class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        
    def compute_region(self, integralImage):
        return integralImage[self.y+self.height][self.x+self.width] + integralImage[self.y][self.x] - integralImage[self.y+self.height][self.x] - integralImage[self.y][self.x+self.width]

class HaarFeature:
    def __init__(self, pos, neg):
        self.positive_regions = pos
        self.negative_regions = neg

    def compute_feature(self, integralImage):
        for rect in self.positive_regions:
            positiveSum = sum([rect.compute_region(integralImage)])
        for rect in self.negative_regions:
            negativeSum = sum([rect.compute_region(integralImage)])
        return  (positiveSum - negativeSum)

class ViolaAndJones:
    def __init__(self, T):
        self.T = T
        self.alphas = []
        self.clfs = []
    
    def build_features(self,imageWidth, imageHeight):
        features = []
        for windowWidth in range(1, imageWidth + 1):
            WW = np.array([windowWidth, 2 * windowWidth, 3 * windowWidth])
            for windowHeight in range(1, imageHeight + 1):
                WH = np.array([windowHeight, 2 * windowHeight, 3 * windowHeight])
                x = 0
                while x + WW[0] < imageWidth:
                    y = 0
                    while y + WH[0] < imageHeight:

                        # (A) 2 Rectangles Horizontal 
                        left = Rectangle(x, y, WW[0], WH[0])
                        right1 = Rectangle(x + WW[0], y, WW[0], WH[0])
                        if x + WW[1] < imageWidth:
                            features.append(HaarFeature([left], [right1]))

                        # (B) 2 Rectangles Vertical
                        bottom = Rectangle(x, y + WH[0], WW[0], WH[0])
                        if y + WH[1] < imageHeight:
                            features.append(HaarFeature([bottom], [left]))

                        # (C) 3 Rectagles Horizontal
                        right2 = Rectangle(x + WW[1], y, WW[0], WH[0])
                        if x + WW[2] < imageWidth:
                            features.append(HaarFeature([left, right2], [right1]))

                        # (C) 3 Rectagles Vertical
                        bottom2 = Rectangle(x, y + WH[1], WW[0], WH[0])
                        if y + WH[2] < imageHeight:
                            features.append(HaarFeature([left, bottom2], [bottom]))

                        # (D) 4 Rectagles
                        bottomRight = Rectangle(x + WW[0], y + WH[0], WW[0], WH[0])
                        if x + WW[1] < imageWidth and y + WH[1] < imageHeight:
                            features.append(HaarFeature([left, bottomRight], [bottom, right1]))
                            
                        y = y + 1
                    x = x + 1
        return features
    
    def apply_features(self, features, training_data):
        X = np.zeros((len(features), len(training_data)))
        Y = []
        for tup in training_data:
            Y.append(tup[1])
        Y = np.asarray(Y)
        for i, feature in enumerate(features):
            for j,image in enumerate(training_data):
                X[i][j] = feature.compute_feature(image[0])
        return X, Y
    
    def select_best(self, classifiers, weights, training_data):
        best_clf = None
        best_accuracy = None
        best_error = math.inf
        for clf in classifiers:
            error = 0
            accuracy = []
            for i in range(len(weights)):
                correct = abs(clf.classify(training_data[i][0]) - training_data[i][1])
                accuracy.append(correct)
                error = error + weights[i] * correct
            error = error / len(weights)
            if error < best_error:
                best_accuracy = accuracy
                best_clf = clf
                best_error = error
        return best_clf, best_error, best_accuracy
    
    def train(self, X, y, features, weights):
        total_positive = 0
        total_negative = 0
        for i in range(y.shape[0]):
            if y[i] == 1:
                total_positive = total_positive + weights[i]
            else:
                total_negative = total_negative + weights[i]
        classifiers = []
        for index, feature in enumerate(X):
            sortedIndices = np.argsort(np.asarray(feature))
            ySorted = y[sortedIndices]
            featureSorted = feature[sortedIndices]
            weightsSorted = weights[sortedIndices]
            applied_feature = zip(weightsSorted,featureSorted,ySorted)
            seen_positive, seen_negative, positive_seen_weights, negative_seen_weights = 0,0,0,0
            min_error, best_feature, best_threshold, best_polarity = math.inf, None, None, None
            for w, f, label in applied_feature:
                error = min(negative_seen_weights + total_positive - positive_seen_weights, positive_seen_weights + total_negative - negative_seen_weights)
                if error < min_error:
                    min_error = error
                    best_threshold = f
                    best_feature = features[index]
                    if seen_positive < seen_negative: 
                        best_polarity = -1 
                    else:
                        best_polarity = 1
                if label == 0:
                    negative_seen_weights = negative_seen_weights + w
                    seen_negative = seen_negative + 1
                else:
                    positive_seen_weights = positive_seen_weights + w
                    seen_positive = seen_positive + 1
            clf = WeakClassifier(best_feature, best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers
        
    def pipeline(self, data, positiveSize):
        negativeSize = len(data) - positiveSize
        trainingData = []
        weights = np.ones(len(data))
        for i in range(len(data)):
            trainingData.append((integral_image(data[i][0]), data[i][1]))
            if data[i][1] == 0:
                weights[i] = 1.0 / (2.0 * negativeSize)
            else:
                weights[i] = 1.0 / (2.0 * positiveSize)
                
        features = self.build_features(trainingData[0][0].shape[0],trainingData[0][0].shape[1])
        X, Y = self.apply_features(features, trainingData)
        features = np.asarray(features)
        self.adaboost(weights,X,Y,features,trainingData)

    def adaboost(self,weights,X,Y,features,trainingData):
        for _ in range(self.T):
            normWeights = np.linalg.norm(weights)
            weights = weights / normWeights
            weak_classifiers = self.train(X, Y, features, weights)
            clf, error, accuracy = self.select_best(weak_classifiers, weights, trainingData)
            self.clfs.append(clf)
            beta = error / (1.0 - error)
            self.alphas.append(np.log(1.0/beta))
            weights = np.asarray(weights)
            accuracy = np.asarray(accuracy)
            weights = weights * (beta ** (1 - accuracy))       
            
    def classify(self, image):
        total = 0
        ii = integral_image(image)
        for i in range(len(self.clfs)):
            total = total + self.alphas[i] * self.clfs[i].classify(ii)
        if total >= 0.5 * sum(self.alphas):
            return 1  
        else:
            return 0  
    
    def save(self, filename):
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)
    @staticmethod
    def load(filename):
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)

class WeakClassifier:
    def __init__(self, haar_feature, threshold, polarity):
        self.haar_feature = haar_feature
        self.threshold = threshold
        self.polarity = polarity
    def classify(self, x):
        feature = self.haar_feature.compute_feature(x)
        x = self.polarity * feature
        y = self.polarity * self.threshold
        if x >= y :
            return 0 
        else:
            return 1

class CascadeClassifier():
    def __init__(self, layers):
        self.layers = layers
        self.clfs = []
    def non_max_suppression(self,boxes, threshold):
	
        boxes = boxes.astype("float")
        coordinates = np.asarray([boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]])
        outputBoxesIndices = []
        indices = np.argsort(coordinates[3])
        area = (coordinates[2] - coordinates[0] + 1) * (coordinates[3] - coordinates[1] + 1)
        while indices.shape[0] > 0:
            boxIndex = indices[-1]
            outputBoxesIndices.append(boxIndex)
            overLapBox = []
            overLapBox.append(np.maximum(coordinates[0][boxIndex], coordinates[0][indices[0:-1]]))
            overLapBox.append(np.maximum(coordinates[1][boxIndex], coordinates[1][indices[0:-1]]))
            overLapBox.append(np.minimum(coordinates[2][boxIndex], coordinates[2][indices[0:-1]]))
            overLapBox.append(np.minimum(coordinates[3][boxIndex], coordinates[3][indices[0:-1]]))
            w = np.where(overLapBox[2] - overLapBox[0] + 1 > 0, overLapBox[2] - overLapBox[0] + 1,0)
            h = np.where(overLapBox[3] - overLapBox[1] + 1 > 0, overLapBox[3] - overLapBox[1] + 1,0)
            overLapValue = (w * h) / area[indices[0:-1]]
            indices = np.delete(indices, np.concatenate(([indices.shape[0] - 1],np.where(overLapValue > threshold)[0])))

        return boxes[outputBoxesIndices].astype("int")

    def get_face_bounding_box(self,input_img, lower_window_range = 19, upper_window_range = 220, window_steps = 5, pixel_steps = 5,ratio = (1/1.3)):
        boxes = [] # (x1,y1,x2,y2)
        for window_height in range(lower_window_range,upper_window_range,window_steps):
            window_width = int((ratio)*window_height)
            number_of_col_iters = input_img.shape[1] - window_width +1
            number_of_row_iters = input_img.shape[0] - window_height +1
            # for i in range(80,240,pixel_steps):
            #     for j in range(30,210,pixel_steps):
            # for i in range(160,480,pixel_steps):
            #     for j in range(60,420,pixel_steps):
            for i in range(0,number_of_row_iters,pixel_steps):
                for j in range(0,number_of_col_iters,pixel_steps):
                    window = input_img[ i:(i+window_height) , (j):(j+window_width)]
                    predicted_test_img = self.classify(cv2.resize(window,(19,19)))
                    if(predicted_test_img):
                        boxes.append((j,i,j+window_width,i+window_height))
        if(len(boxes) == 1):
            return boxes[0]
        if(len(boxes) != 0):
            boxes = np.asarray(boxes)
            nmb = self.non_max_suppression(boxes,0.3)
            if(nmb.shape[0]>1):
                fix = []
                for box in nmb:
                    mid = [(box[0]+box[2])//2,(box[1]+box[3])//2]
                    fix.append(abs((mid[0]-160)) + abs((mid[1]-120)))
                return nmb[np.argmin(np.asarray(fix))]
            return nmb[0]
        return [0, 0, 0, 0]
    
    def classify(self, image):
        for clf in self.clfs:
            if clf.classify(image) == 0:
                return 0
        return 1
    def save(self, filename):
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)
    @staticmethod
    def load(filename):
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)