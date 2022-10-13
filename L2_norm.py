#Importing modules
import cv2
import numpy as np
import glob
import math

# Training function
def Training_function():
    crabs = [cv2.imread(file) for file in glob.glob('C:/Users/Maryam/PycharmProjects/KNN/animals/crab/*.jpg')]
    dolphins = [cv2.imread(file) for file in glob.glob('C:/Users/Maryam/PycharmProjects/KNN/animals/dolphin/*.jpg')]

    reshape_crab = []
    reshape_dolphin = []
    for a in range(10):
        crabs[a].resize((32,32,3))
        reshape_crab.append(np.reshape(crabs[a],[3072,1], order = "F"))

    for b in range(10):
        dolphins[b].resize((32, 32, 3))
        reshape_dolphin.append(np.reshape(dolphins[b],[3072,1], order = "F"))

    return reshape_crab, reshape_dolphin


def Testing_function(k:int):
    c1, c2 = Training_function()


    testing_image = cv2.imread('C:/Users/Maryam/PycharmProjects/KNN/animals/Test/c4.jpg')
    showing_image = cv2.imread('C:/Users/Maryam/PycharmProjects/KNN/animals/Test/c4.jpg')
    testing_image.resize((32,32,3))

    image = np.reshape(testing_image, -1, order="F")


    distances = []
    labels = []

    for a in range(10):
        distances.append(L2_Norm(image, c1[a]))
        labels.append(1)

    for a in range(10):
        distances.append(L2_Norm(image, c2[a]))
        labels.append(2)

    Distances_S = np.sort(distances)

    ind = []
    NN_vote = []
    for a in range(k):
        ind.append(np.where(distances == Distances_S[a]))

    for b in range(k):
        NN_vote.append(labels[ind[b][0][0]])

    count1 = 0
    count2 = 0

    print("Nearest Neighbour Vote = ", NN_vote)

    for a in range(k):
        if NN_vote[a] == 1:
            count1 = count1 + 1
        else:
            count2 = count2 + 1

    if count1 > count2:
        image = cv2.putText(showing_image, 'Crab', (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)
        print("Testing Image belongs to Crab class.")
    else:
        image = cv2.putText(showing_image, 'Dolphin', (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)
        print("Testing Image belongs to Dolphin class.")

    cv2.imshow("Testing Image",image)
    cv2.waitKey(0)


def L2_Norm(img_train, img_test):
    d = 0.0
    for a in range(len(img_train)):
        Eucledian_Ditance = d + (img_test[a] - img_train[a])*2

    return math.sqrt(Eucledian_Ditance)

def print_hi(name):

    Testing_function(5)




if __name__ == '__main__':
    print_hi('PyCharm')