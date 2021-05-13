# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 04:20:28 2020

@author: adars
"""

import numpy as np
from math import log
import matplotlib.pyplot as plt
import cv2
import time
import os
import glob
from sklearn.metrics import roc_curve, auc
window_size = 16

class Haar(object):
    def __init__(self, type, feature, size, shape, start):
        self.type=type
        self.feature = cv2.resize(feature, (shape[1]*size, shape[0]*size), interpolation=cv2.INTER_NEAREST)
        self.start = start
        self.size=size
        self.shape = shape
        


class WeakClassifier(object):
    def __init__(self, haar, theta, sign, weight):
        self.haar = haar
        self.theta = theta
        self.sign = sign
        self.weight = weight


def blockSum(integral_ii, w, h, w_kernel_size, h_kernel_size):
    endw =int( w + w_kernel_size - 1)
    endh =  int (h + h_kernel_size - 1)
    sum = integral_ii[endh,endw]
    if (w > 0):
        sum -= integral_ii[int(endh),int( w - 1)]
    if (h > 0):
        sum -= integral_ii[int(h - 1),int( endw)]
    if (h > 0 and w > 0):
        sum += integral_ii[int(h - 1),int( w - 1)]
    return sum

def sub_integral(subwindow):
    integral_ii = np.zeros(subwindow.shape)

    for r in range(subwindow.shape[0]):
        for c in range(subwindow.shape[1]):
            greyIntegralVal = subwindow[r,c]

            if (r - 1 >= 0 and c - 1 >= 0):
                greyIntegralVal -= integral_ii[r-1,c-1]

            if (r - 1 >= 0):
                greyIntegralVal += integral_ii[r-1,c]

            if (c - 1 >= 0):
                greyIntegralVal += integral_ii[r,c-1]

            integral_ii[r,c]  = greyIntegralVal

    return integral_ii


def get_haar_score_fast(haar, subwindow, sub_integral):
    r = haar.start[0]
    c = haar.start[1]
    size_w = haar.feature.shape[1]
    size_h = haar.feature.shape[0]

    if size_w==2 and size_h==2:
        # For simple feature, not using integral
        if haar.type==1:
            return subwindow[r,c]+subwindow[r+1,c]-subwindow[r,c+1]-subwindow[r+1,c+1]
        elif haar.type==2:
            return subwindow[r,c]+subwindow[r,c+1]-subwindow[r+1,c]-subwindow[r+1,c+1]
        elif haar.type==5:
            return subwindow[r,c]+subwindow[r+1,c+1]-subwindow[r,c+1]-subwindow[r+1,c]
        elif haar.type==6:
            return subwindow[r+1,c]+subwindow[r,c+1]-subwindow[r,c]-subwindow[r+1,c+1]
    else:
        # For complex features, using integral to reduce array references
        if haar.type==1:
            return blockSum(sub_integral, c, r, size_w / 2, size_h) - blockSum(sub_integral, c + (size_w / 2), r, size_w / 2, size_h)
        elif haar.type==2:
            return blockSum(sub_integral, c, r, size_w, size_h / 2) - blockSum(sub_integral, c, r + (size_h / 2), size_w, size_h / 2)
        elif haar.type==3:
            return blockSum(sub_integral, c, r, size_w, size_h) - 2 * blockSum(sub_integral, c + (size_w / 3), r, size_w / 3, size_h)
        elif haar.type==4:
            return blockSum(sub_integral, c, r, size_w, size_h) - 2 * blockSum(sub_integral, c, r + (size_h / 3), size_w, size_h / 3)
        elif haar.type==5:
            return blockSum(sub_integral, c, r, size_w, size_h) - 2 * (blockSum(sub_integral, c + (size_w / 2), r ,size_w / 2, size_h / 2) + blockSum(sub_integral, c, r + (size_h / 2), size_w / 2, size_h / 2))
        elif haar.type==6:
            return blockSum(sub_integral, c, r, size_w, size_h) - 2 * (blockSum(sub_integral, c, r, size_w / 2, size_h / 2) + blockSum(sub_integral, c + (size_w / 2), r + (size_h / 2), size_w / 2, size_h / 2))
        elif haar.type==7:
            if (size_w == 3):
                return blockSum(sub_integral, c,r, size_w, size_h) - 2 * subwindow[r + 1, c + 1]
            else:
                return blockSum(sub_integral, c, r, size_w, size_h) - 2 * blockSum(sub_integral, c + (size_w / 3), r + (size_h / 3), size_w / 3, size_h / 3)



def feature_weighted_error_rate(actual, predicted, weights):
    return sum(weights*(np.not_equal(actual, predicted)))


def predict(score, classifier):
    if score<classifier.theta:
        return -classifier.sign
    return classifier.sign


def display(fovea):
    plt.imshow(fovea, interpolation='nearest')
    plt.show()



os.getcwd()
path_face = glob.glob('.\data_set_f\*.bmp')
path_nonface= glob.glob('.\data_set_nf\*.bmp')
#f_list
positive_samples=[]
#nf_list
negative_samples=[]
for i in path_face[0:50]:   #change here to increase the Positives samples ( face data)
    positive_samples.append(np.divide(np.asarray(cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2GRAY),dtype=np.float64),255))

for i in path_nonface[0:100]:       #change here to increase the number of Negative Sample ( Non face Data)
    negative_samples.append(np.divide(np.asarray(cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2GRAY),dtype=np.float64),255))
# taking Rotated faces to increase the total positives 
positive_samples_rotated = [np.rot90(np.rot90(np.rot90(el))) for el in positive_samples]
positive_samples = positive_samples + positive_samples_rotated

# no need to rotate the negatives (won't make difference)
#negative_samples_rotated = [np.rot90(np.rot90(np.rot90(el))) for el in negative_samples]
#negative_samples = negative_samples + negative_samples_rotated

np.random.shuffle(positive_samples)
np.random.shuffle(negative_samples)

# number of positive sample and negatives sample

print("Total num of samples at our disposal:")
print("Positives: " + str(len(positive_samples)))
print("Negatives: " + str(len(negative_samples)))

# splitting Data so that 85 % is training and rest 15 % is testing
split = 0.85
pos_split = int(len(positive_samples)*split)
neg_split = int(len(negative_samples)*split)

training_set = positive_samples[0:pos_split] + negative_samples[0:neg_split]
testing_set = positive_samples[pos_split:] + negative_samples[neg_split:]

training_integrals = [sub_integral(i) for i in training_set]
testing_integrals = [sub_integral(i) for i in testing_set]

nrPos = pos_split
nrNeg = neg_split
nrPos_test = len(positive_samples)-nrPos
nrNeg_test = len(negative_samples)-nrNeg

training_labels = [1]*nrPos + [-1]*nrNeg
testing_labels = [1]*nrPos_test + [-1]*nrNeg_test

print("For training")
print("Positives: "+str(nrPos))
print("Negatives: "+str(nrNeg))

print("For testing")
print("Positives: "+str(nrPos_test))
print("Negatives: "+str(nrNeg_test))


'''------- Generate many haar features------'''
features_start=[]

'''-------- Define haar feature types-------------'''
haar1 = np.array([1, -1,
                  1, -1])
haar1.shape = (2,2)

haar2 = np.array([1, 1,
                  -1, -1])
haar2.shape = (2,2)

haar3 = np.array([1, -1, 1,
                  1, -1, 1])
haar3.shape = (2,3)

haar4 = np.array([1, 1,
                  -1, -1,
                  1, 1])
haar4.shape = (3,2)

haar5 = np.array([1, -1,
                  -1, 1])
haar5.shape = (2,2)

haar6 = np.array([-1, 1,
                  1, -1])
haar6.shape = (2,2)

haar7 = np.array([1, 1, 1,
                  1, -1, 1,
                  1, 1, 1])
haar7.shape = (3,3)


'''------ Define many sizes for all feature types--------'''

haar_feature_types=[haar1,haar2,haar3,haar4,haar5,haar6, haar7]
for f in range(len(haar_feature_types)):
    shape = haar_feature_types[f].shape
    if 3 in haar_feature_types[f].shape:
        max_size=4
    else:
        max_size=7

    for s in range(1, max_size+1):
        features_start.append(Haar(f+1, haar_feature_types[f], s, shape, (0,0)))

features = []
for j in features_start:
        # Get all posible starting locations for this feature
        start_positions = []
        space = (window_size-j.shape[0]*j.size, window_size-j.shape[1]*j.size)
        for k in range(space[0]+1):
            for l in range(space[1]+1):
                start_positions.append((k, l))

        for loc in start_positions:
            features.append(Haar(j.type, j.feature, j.size, j.shape, loc))


#print("hello")
features = list(features)
temp1 = len(set(features))
temp2 = len(features)
print(temp1)
#print(temp2)

feature_weights=[]
weak_classifires = []


np.random.shuffle(features)


errors = []
scores = []
thetas = []
polarities = []
'''------ For every feature, find best threshold and compute corresponding weighted error-------'''
for j in features:
    avgPosScore = 0.0
    avgNegScore = 0.0
    # Apply feature to each image and get threshold for current feature (current location)
    for i in range(len(training_set)):
        score=get_haar_score_fast(j, training_set[i], training_integrals[i])
        scores.append(score)

        if training_labels[i]==1:
            avgPosScore += score
        else:
            avgNegScore += score

    avgPosScore = avgPosScore / nrPos
    avgNegScore = avgNegScore / nrNeg
    if avgPosScore>avgNegScore:
        polarity = 1
    else:
        polarity = -1
    polarities.append(polarity)

    # Optimal theta found
    theta = (avgPosScore + avgNegScore) / 2
    thetas.append(theta)


'''----------Cascade Creation-------------'''

F_target = 0.001
f = 0.5

F_i = 1
#i = 0


cascade = []
start_time = time.time()
# initialising weights so that sum is 1
image_weights = [1.0/(2*nrPos)]*nrPos + [1.0/(2*nrNeg)]*nrNeg

show_stuff = False

while F_i > F_target:
    #i += 1
    ## Train classifier for stage i

    #best_feature_index = 0
    best_weak_classifier = 0
    lowest_error = float("inf")

    # image_weights = [1.0/(2*nrNeg)]*nrNeg + [1.0/(2*nrPos)]*nrPos
    total = sum(image_weights)
    image_weights = [w / total for w in image_weights]
    TP=0
    TN=0

    f_i = 1
    cycle = 0

    while (TP/nrPos<0.5) and (TN/nrNeg<0.5):
        total = sum(image_weights)
        if total != 1:
            image_weights = [w / total for w in image_weights]

        print(" ")
        errors = []
        # For every feature, find best threshold and compute corresponding weighted error
        loop_cnt = 0
        inner_loop_cnt = 0
        for j in features:

            # Create classifier object
            w_classif = WeakClassifier(j, thetas[loop_cnt], polarities[loop_cnt], 0)

            # Compute weighted error
            predicted = []
            for sample in range(len(training_set)):
                # Get predictions of all samples
                score=scores[inner_loop_cnt]
                predicted.append(predict(score, w_classif))
                inner_loop_cnt += 1

            weighted_error=feature_weighted_error_rate(training_labels, predicted, image_weights)
            errors.append(weighted_error)

            # Look for the lowest error and keeping track of the corresponding classifier
            if weighted_error<lowest_error:
                lowest_error = weighted_error
                best_weak_classifier = w_classif
                #best_feature_index = features.index(j)

            loop_cnt+=1
            #print('foooooooooooo')
        # print("Best feature index: "+str(best_feature_index))

        if show_stuff:
            plt.plot(errors)
            plt.show()

        ## Choose weak classifier with lowest error ##
        beta_t = lowest_error/(1-lowest_error)

        if beta_t==0:
            inverted_weighth = 0
        else:
            inverted_weighth = log(1/beta_t)
        best_weak_classifier.weight = inverted_weighth

        ## Update weights and evaluate current weak classifier ##
        predicted=[]
        scores_debug = []
        for sample in range(len(training_set)):
            # Get weighted classification error
            score=get_haar_score_fast(best_weak_classifier.haar, training_set[sample], training_integrals[sample])
            scores_debug.append(score)
            predicted.append(predict(score, best_weak_classifier))

        FP = 0.0
        FN = 0.0
        TP = 0.0
        TN = 0.0
        visual_predicted = []
        for k in range(len(image_weights)):
            # if sample is correctly classified

            if training_labels[k] == 1 and predicted[k] == -1:
                FN += 1
            if training_labels[k] == -1 and predicted[k] == 1:
                FP += 1

            # Update image weights
            if training_labels[k]==predicted[k]:
                image_weights[k] = image_weights[k]*beta_t
                if predicted[k] == 1:
                    TP += 1
                if predicted[k] == -1:
                    TN += 1

            if predicted[k]==-1:
                visual_predicted.append('r')
            else:
                visual_predicted.append('g')

        # Evaluate f_i
        f_i = (FP/(2*nrNeg))+(FN/(2*nrPos))
        print("f_i: " + str(f_i))

        print("TP, TN, FP, FN for the current weak classifier:")
        print(TP/nrPos, TN/nrNeg, FP/nrNeg, FN/nrPos)

        # Visualize the performace of weak classifier for training samples
        if show_stuff:
            plt.scatter(range(nrPos+nrNeg), scores_debug, c = visual_predicted)
            plt.vlines(nrPos,min(scores_debug),max(scores_debug))
            plt.plot(range(nrPos+nrNeg), [best_weak_classifier.theta]*(nrPos+nrNeg))
            plt.xlim(0,nrPos+nrNeg)
            plt.show()

        print("Threshold of the best feature: "+str(best_weak_classifier.theta))

        cycle += 1

    cascade.append(best_weak_classifier)

    print(len(features))

    print(best_weak_classifier.haar.feature)

    strong_FP = 0.0
    strong_FN = 0.0

    cascade_scores = []
    cascade_colors_predicted = []
    for l in range(len(training_set)):
        strong_score = 0.0
        for w_class in cascade:
            strong_score += w_class.weight * predict(get_haar_score_fast(w_class.haar, training_set[l], training_integrals[l]), w_class)
        cascade_scores.append(strong_score)
        clas = np.sign(strong_score)
        if clas==-1:
            cascade_colors_predicted.append('r')
        else:
            cascade_colors_predicted.append('g')

        if training_labels[l] == 1 and clas == -1:
            strong_FN += 1
        if training_labels[l] == -1 and clas == 1:
            strong_FP += 1

    # Visualize the performace of the cascade on training samples
    if show_stuff:
        plt.scatter(range(nrPos+nrNeg), cascade_scores, c = cascade_colors_predicted)
        plt.vlines(nrPos,min(cascade_scores),max(cascade_scores))
        plt.plot(range(nrPos+nrNeg), [0]*(nrPos+nrNeg))
        plt.xlim(0,nrPos+nrNeg)
        plt.show()

    F_i = (strong_FP/(2*nrNeg))+(strong_FN/(2*nrPos))
    print("F_i: " + str(F_i))
    print("Cascade size: "+str(len(cascade)))

print("--- %s seconds ---" % (time.time() - start_time))


print("Now running cascade on the testing set")

FP_test = 0.0
FN_test = 0.0
TP_test = 0.0
TN_test = 0.0

scores = []
print("Cascade:")
print(cascade)

f_cnt=1
for i in cascade:
    print(" ")

    print("// Compute "+str(f_cnt)+" feature score")
    print("vote = f_vote(greyIntegral, "+str(i.haar.type)+", w, h, scale*"+
          str(i.haar.feature.shape[1])+", scale*"+str(i.haar.feature.shape[0])+", scale*"+str(i.haar.start[1])+", scale*"+str(i.haar.start[0])+
          ", scaleTh*"+str(int(round(i.theta)))+", "+str(i.weight*i.sign)+", fovea);")
    print("cascade_score += vote;")

    f_cnt+=1

save = False

for t in range(len(testing_set)):
    strong_score = 0.0
    for w_class in cascade:
        #print("Loc: " +str(w_class.haar.start))
        strong_score += w_class.weight * predict(get_haar_score_fast(w_class.haar, testing_set[t], testing_integrals[t]), w_class)
    clas = np.sign(strong_score)
    scores.append(strong_score)

    if testing_labels[t] == 1 and clas == -1:
        FN_test += 1
        if save:
            plt.imshow(testing_set[t], interpolation='nearest')
            plt.savefig("FN/"+str(t)+".jpg")
    if testing_labels[t] == -1 and clas == 1:
        FP_test += 1
        if save:
            plt.imshow(testing_set[t], interpolation='nearest')
            plt.savefig("FP/"+str(t)+".jpg")
    if testing_labels[t] == 1 and clas == 1:
        TP_test += 1
        if save:
            plt.imshow(testing_set[t], interpolation='nearest')
            plt.savefig("TP/"+str(t)+".jpg")
    if testing_labels[t] == -1 and clas == -1:
        TN_test += 1
        if save:
            plt.imshow(testing_set[t], interpolation='nearest')
            plt.savefig("TN/"+str(t)+".jpg")


print((FP_test/(2*nrNeg_test))+(FN_test/(2*nrPos_test)))
print(FP_test)
print(FN_test)
plt.plot(range(nrPos_test), scores[0:nrPos_test], 'go')
plt.plot(range(nrPos_test,nrPos_test+nrNeg_test), scores[nrPos_test:], 'ro')
#xyz=len(cascade)
#plt.plot(range(xyz), [0]*xyz)
plt.plot(range(nrNeg_test), [0]*nrNeg_test)
plt.show()

print("TP, TN, FP, FN for the cascade classifier:")
print(TP_test/nrPos_test, TN_test/nrNeg_test, FP_test/nrNeg_test, FN_test/nrPos_test)

'''--------------------Computing Error Rate and Accuracy--------------------'''

error_rate=(FP_test+FN_test)/(nrPos_test+nrNeg_test)
accuracy=1-error_rate
print("Error Rate and Accuracy: ")
print(error_rate, accuracy)

'''------------- Plotting the final ROC curve--------------------'''
 
roc_predictions=[]
roc_predictions.append(np.sign(scores))
fpr = []
tpr = []
roc_auc = []
fpr, tpr, _ = roc_curve(testing_labels,roc_predictions[0])
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
