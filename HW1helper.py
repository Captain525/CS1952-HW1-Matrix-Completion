import numpy as np
import cupy as cp
import time as time
from numpy.linalg import svd

def loadPredictionsFromFile(fileName, q):
    """
    Load predictions from a file. 
    """
    with open(fileName, 'r') as f:
        #this first line contains n, m, and k. 
        fileLines = f.readlines()
        predictionList = [float(line.split()[0]) for line in fileLines]
        predictionArray = np.array(predictionList, np.float32)
        assert(predictionArray.shape == (q, ))
        return predictionArray
def checkPredictionsDistance(predictionArray1, predictionArray2):
    """
    Check how far given predictions are from eachother. 
    """
    dist = cp.square(cp.linalg.norm(predictionArray1-predictionArray2))/predictionArray1.shape[0]
    print("dist between two is: ", dist)
    return dist

def checkPredictionsMatchWithActualOrdering(q):
    """
    Check my new predictions match with the old ones, are equally good. 
    """
    submittedPredictions = loadPredictionsFromFile("mat_comp_ans.txt", q)
    roundedPredictions= loadPredictionsFromFile("predictionsRounded.txt", q)
    dist = checkPredictionsDistance(submittedPredictions, roundedPredictions)
    threshold = .2
    assert(dist<threshold)
    return
def getBooleanMatrixM(dataMatrix, n, m):
    """
    Goal: Get a boolean matrix of size nxm which represents which entries we have data of 
    and which we don't. 

    Can pass in a subset of the whole data matrix if we want to save some test data for validation. 

    Input data in the form of numTrainingExamples x 3, where first column is i(user i), second column is j(movie j) and third is the rating value for that spot.
    User starts at 1, movie starts at 1. 
    """
    M = np.zeros(shape = (n,m))
    indices = dataMatrix[:, 0:2].astype(np.int32)
    ratings = dataMatrix[:, 2]
    #use the indices pairs as input to M, and assign the corresponding rating to that point in the matrix. 
    #subtract one bc indices start at 1. 
    M[indices[:,0]-1, indices[:,1]-1] = ratings
    booleanArray  = M.astype(bool)
  
    return M, booleanArray

def importDataFromFile(fileName):
    """
    This method imports data from the mat_comp file, and organizes it into 
    a training ratingMatrix, and a test predictMatrix, also returning n and m. 
    """
    with open(fileName, 'r') as f:
        #this first line contains n, m, and k. 
        fileLines = f.readlines()
        integers = fileLines[0]
        listNums = integers.split()
        nums = [int(i) for i in listNums]
        n = nums[0]
        m = nums[1]
        k = nums[2]
        print(n, m, k)
        ratings = fileLines[1:k+1]
        assert(len(ratings) == k)
        ratingList = [[float(line.split()[i]) for i in range(3)] for line in ratings]
        #first two elements of each row are ints i and j, third column is decimals like 3.5. 
        ratingMatrix = np.array(ratingList, dtype = float)
        assert(ratingMatrix.shape == (k, 3))
        #get the q value from this line. The entire line should just be q, so 
        #calling int will get the number. 
        qString = int(fileLines[k+1])
        q = int(qString)
        #get the things we need to predict.  
        predictLines = fileLines[k+2:]
        assert(len(predictLines) == q)
        predictionList = [[int(line.split()[i]) for i in range(2)]for line in predictLines]
        predictMatrix = np.array(predictionList, dtype = int)
        assert(predictMatrix.shape == (q, 2))
        assert(predictMatrix[0, 0] == int(predictLines[0].split()[0]))
        assert(predictMatrix[-1, 1] == int(predictLines[-1].split()[1]))
        #can get k and q from sizes of rating and predict matrix. 
        return (n,m, ratingMatrix, predictMatrix)

def readyData(n,m , ratingMatrix):
    """
    Prepare data for training by splitting it up into training, validation, and test. 
    """
    k = ratingMatrix.shape[0]
    #FORGOT THIS BEFORE. MUCH BETTER WITH SHUFFLE, so that each group are sharing each part of the matrix. 
    np.random.shuffle(ratingMatrix)
    trainPercent = .8
    validationPercent = .1
    testPercent = 1 - (trainPercent + validationPercent)
    first = int(.8*k)
    second = int((trainPercent + validationPercent)*k)
    #train val test split of the rating matrix. 
    trainRating = ratingMatrix[0:first]
    validationRating = ratingMatrix[first:second]
    testRating = ratingMatrix[second:]
    booleanStart = time.time()
    Mtrain, Btrain = getBooleanMatrixM(trainRating, n, m)
    booleanEnd = time.time()
    booleanTime = booleanEnd-booleanStart
    print("booleanTime: ", booleanTime)
    #gets the boolean matrices for the validation and test here. 
    Mval, Bval = getBooleanMatrixM(validationRating, n, m)
    Mtest,Btest = getBooleanMatrixM(testRating, n, m)
    return [(Mtrain, Btrain), (Mval, Bval), (Mtest, Btest)]
def initialize(n, m, r, Mtrain):
    """
    Initialize the matrices we wish to learn. 
    Could do this in many different ways, this could also help with learning. 

    Just does normal 0,1 for now. 
    """
    X = np.random.normal(0,1, (n,r))
    Y = np.random.normal(0, 1, (m,r))
    """
    assert(Mtrain.shape == (n,m))
    startSVD = time.time()
    U, s, Vt = cp.linalg.svd(cp.array(Mtrain))
    endSVD = time.time()
    print("SVD TIME: ", endSVD - startSVD)
    X = U[:, 0:r]
    print(X.shape)
    assert(X.shape == (n,r))
    sp = cp.diag(s[0:r])
    print(sp.shape)
    X = X@sp

    Y = cp.transpose(Vt[0:r, :])
    print(Y.shape)
    X = U@s
    """
    assert(X.shape == (n, r))
    assert(Y.shape == (m, r))
    return X,Y

def checkGradient(X,Y,M,B, isItX,grad):
    """
    Does gradient computation without the "B" in order to make sure the math is right. 
    Don't call this in final submission. 
    """
    matrixTerm = M - (X@Y.T)
    n = X.shape[0]
    m = Y.shape[0]
    r = X.shape[1]
    assert(Y.shape[1] == r)
    if(isItX):
        gradManual = np.zeros(X.shape)
        for k in range(0, n):
            for l in range(0, r):
                insideFirst = matrixTerm[k, :]
                #if k,j is in omega, then this term is equal to Y, if not its 0. 
                insideSecond = B[k,:]*Y[:, l]
                gradManual[k,l] = -2*np.dot(insideFirst, insideSecond)
    else:
        gradManual = np.zeros(Y.shape)
        for k in range(0, m):
            for l in range(0,r):
                insideFirst = matrixTerm[:,k]
                insideSecond = B[:, k]*X[:, l]
                gradManual[k,l] = -2*np.dot(insideFirst, insideSecond)
    difference = np.abs(grad - gradManual)
    print("difference is: ", difference)
    avgDifference = np.mean(difference)
    stdDiff = np.std(difference)
    print("avg diff: ", avgDifference, "\n")
    print("stdDiff: ", stdDiff, "\n")
    threshold = .01
    return np.all(difference<threshold)


def checkPredictions(predictions, Mfinal, predictData):
    """
    Check if hte way i did the previous prediction method works. 
    """
    slowPredictions = cp.zeros(shape = (predictions.shape))
    for point in range(predictData.shape[0]):
        i, j = predictData[point, :]
        assert(i>0)
        assert(j>0)
        slowPredictions[point] = Mfinal[i-1, j-1]
    difference = cp.abs(slowPredictions - predictions)
    meanDiff = cp.mean(difference)
    stdDiff = cp.std(difference)
    print("mean diff: {}, std diff: {}".format(meanDiff, stdDiff))
    threshold = .01
    valid = cp.all(difference<threshold)
    return valid