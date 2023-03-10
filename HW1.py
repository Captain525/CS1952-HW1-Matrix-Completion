import numpy as np 
import cupy as cp
from HW1helper import *
from scipy.linalg import svd
#from HW1helper import importDataFromFile
"""
Want to train on completing the matrix, ie M = XY^T. 
Loss function is sum of squared error but only on the examples we have in training data. 
want to hold out some data as training data. 

"""
def runProgram():
    """
    Main method to run the program, doing gradient descent by hand. 
    """
    small = "mat_comp_small"
    big = "mat_comp"
    file_name = big
    
    #Gets the data from the file and loads it into ratingMatrix(stuff with training values) 
    # and predictMatrix(stuff we need to predict).
    n,m, ratingMatrix, predictMatrix = importDataFromFile(file_name)
    #split the data into training, validation, and testing, as well as randomize the order. 
    q = predictMatrix.shape[0]
    checkPredictionsMatchWithActualOrdering(q)
    realAnswers = cp.array(loadPredictionsFromFile("mat_comp_ans.txt", q))
    realAnswersUnrefined = cp.array(loadPredictionsFromFile("mat_comp_ans_unrefined.txt", q))

    #roundedRealAnswers = roundFringePredictions(realAnswers)
    checkPredictionsDistance(realAnswersUnrefined, realAnswers)
    #printToFile(roundedRealAnswers, "mat_comp_ans_rounded.txt")
    (Mtrain, Btrain), (Mval,Bval), (Mtest,Btest) = readyData(n,m,ratingMatrix)
    #pick an r value: I chose r=6
    r = chooseR(n, m, Mtrain, Btrain)
    #initialize X and Y. I just did N(0,1). 
    X,Y = initialize(n,m, r, Mtrain)
    numEpochs = 1000
    learningRate = .00004
    batchSize = 100
    #do gradient descent with the a batch being a number of rows. 
    Xfinal, Yfinal = batchGradientDescent(Mtrain, Mval, X, Y, Btrain, Bval, learningRate, numEpochs, batchSize)
    #get the final loss of the result. 
    testLoss = loss(cp.array(Mtest), cp.array(Btest), Xfinal, Yfinal)
    print("test loss is: ", testLoss)

    Munaltered = Xfinal@Yfinal.T
    MroundFringes = roundMatrixFringes(Munaltered)
    accuracyUnaltered = accuracy(cp.array(Mtest), Munaltered, cp.array(Btest))
    accuracyFringes = accuracy(cp.array(Mtest), MroundFringes, cp.array(Btest))
    print("unaltered accuracy: ", accuracyUnaltered)
    print("Rounded accuracy: ", accuracyFringes)
    predictions, predictionsRounded = predict(Xfinal, Yfinal, cp.array(predictMatrix))
    assert(predictions.shape[0] == q)

    printToFile(predictions, "predictions.txt")
    printToFile(predictionsRounded, "predictionsRounded.txt")
    return


def batchGradientDescent(Mtrain, Mtest, X, Y, Btrain, Btest, learningRate, numEpochs, batchSize):
    """
    Calculates a gradient on batchSize adjacent rows in X or Y. This is much faster than the row based one, but 
    potentially the gradients are less exact. 
    """
    n = X.shape[0]
    m = Y.shape[0]
    X0 = cp.array(X)
    Y0 = cp.array(Y)
    X1 = X0
    Y1 = Y0
    Mtrain = cp.array(Mtrain)
    Mtest = cp.array(Mtest)
    Btrain = cp.array(Btrain)
    Btest = cp.array(Btest)
    #assuming int rounds down. 
    xBatches = int(n/batchSize) + 1
    yBatches = int(m/batchSize)+ 1
    #one group has leftovers, other has leftovers+d. know leftovers<bs. 
    for i in range(0, numEpochs):
        #shuffles the batch order randomly. 
        biggestBatchSize = max(xBatches, yBatches)
        perm = np.arange(0, biggestBatchSize)
        np.random.shuffle(perm)
        startLoop = time.time()
       
        for b in perm:
            #gets a batch. 
            MbatchX = Mtrain[batchSize*b:batchSize*(b+1),:]
            BbatchX = Btrain[batchSize*b:batchSize*(b+1), :]
            MbatchY = Mtrain[:, batchSize*b:batchSize*(b+1)]
            BbatchY = Btrain[:, batchSize*b:batchSize*(b+1)]

            if(b<xBatches):
                Xbatch = X0[batchSize*b:batchSize*(b+1),:]
                xGrad = calculateGradient(Xbatch, Y0, MbatchX, BbatchX, True)
                X1[batchSize*b:batchSize*(b+1), :] = Xbatch - learningRate*xGrad
                X0 = X1
            if(b<yBatches):
                Ybatch = Y0[batchSize*b:batchSize*(b+1), :]
                yGrad = calculateGradient(X1, Ybatch, MbatchY, BbatchY, False)
                Y1[batchSize*b:batchSize*(b+1), :] = Ybatch - learningRate*yGrad
                Y0 = Y1
        endLoop = time.time()
        timeLoop = endLoop-startLoop
        print("loop time: ", timeLoop)
        trainLoss = loss(Mtrain, Btrain, X1, Y1)
        testLoss = loss(Mtest, Btest, X1, Y1)
        print("Epoch {epoch} train loss: {tloss}, test loss: {vloss}".format(epoch=i, tloss = trainLoss, vloss = testLoss))
    return X1, Y1


def calculateGradient(X,Y, M, B, isItX, check=False):
    """
    X - X matrix nxr
    Y - Y matrix m x r
    M - matrix where if i,j in Omega, Mij = Mij, if ij not in omega Mij = 0
    B - Boolean matrix where true if ij in omega, false if not. 
    TO calculate the gradient with respect to X (which should be in Rnxr), 
    we got -2(M-XY^t * B)Y. This B is a way of dealing with the fact we're only summing
    over the ij in Omega. 
    The simpler way of writing this is: 
    df/Dxkl = -2sum_{j s.t (k,j) in Omega}(M_{kj} - XY^t_{kj})Y_{jl}, because for i !=k, 
    the derivative of XY^Tij with respect to X_{kl} is 0. For k = i, the derivative is -Yj,l. 
    Note that if ij isn't in omega, then the terms in the matrix multiplication will be 0. 

    To calculate the gradient with respect to Y, which should be in Rmxr, 
    we got -2(M-XY^T*B)^T Y. This B deals wtih that sum again. 
    Simpler way:
    df/Dykl = -2sum_{i s.t (i,k) in OMega}(M - XY^T)_{ik}X_{il}

    GOT TRUE FOR MY RESULTS. 
    If checkGradient is right, then this method is right. 

    check is an optional parameter which calls checkGradient. 

    adding regularization. 
    MADE REGULARIZATION FASTER BY ADDING IT TO THIS METHOD. 
    """
    doReg = True
    val = X@Y.T
    #coefficient for regularization
    l = 1
    term = (M-(val)*B)
    #do regularization here. 
    if(doReg):
        #If greater than 5
        I = (val>5).astype(int)
        #if less than .5
        J = (val<.5).astype(int)
        #This term basically gets the distance from the boundary as the gradient value, 
        # so if it's farther away it pushes harder. It's actually L2 loss, but doesn't show because we  only look at the gradient. 
        #Note that this term isn't included in our printed loss because I wanted that to reflect the grader's loss. 
        valTerm = val*(I+J)-(5*I + .5*J)
        term = term -l*valTerm
    if(isItX):
        grad = term@Y
        assert(grad.shape == X.shape)
    else:
        grad = term.T @X
        assert(grad.shape == Y.shape)
    if(check):
        #check if gradient computation was correct. 
        assert(checkGradient(X,Y,M,B, isItX, grad))
    return -2*grad
def loss(M, B, X, Y):
    """
    The base loss function is: 
    f(X,Y) = sum_{i,j in omega}(Mij - (XY^T)ij)^2
    We can rewrite this in matrix form however, using B. 
    Note if i,j isn't in omega, by definition of M, Mij = 0. 
    By definition of B, Bij = 0. So, if we plug in XY^T*B, then Mij - (XY^T*B) = 0
    for those which shouldn't be counted which is what we want. 
    Thus, f(X,Y) = ||M - XY^T*B||_F^2 - frobenius norm squared of M-XY^T*B

    We can also customize M and B to only look at certain training examples, and this
    works for both testing and training data. 

    cupy made this method faster, but made the gradient loop slower. 
    It made this go at around .00x seconds, whereas with numpy .8 seconds. 

    However, loop time doubled. 
    So, just cast this method to cp, which makes it run faster. 
    """
    #cast to cp to make this operation faster. 
    numEntries = cp.sum(B)
    mat = M-(X@Y.T)*B
    #this method is WAY faster. However, need to square it somehow. 
    #still faster with these extra things added. 
    loss = cp.square(cp.linalg.norm(mat))/numEntries
    return loss

def predict(X, Y, predictData):
    """
    Calculate predictions for the data. 
    Predictions: q x 2, where first column is user second column is movie. Indexed starting at one. 
    To predict, just pick the ERM, ie the value of XY^T at i,j
    """
    Mfinal = X@Y.T
    predictions = Mfinal[predictData[:, 0]-1, predictData[:, 1]-1]
    predictionsRounded = roundFringePredictions(predictions)
    return predictions, predictionsRounded

def accuracy(M, M_hat, B):
    """
    Calculate the loss of a specific output matrix M_hat with M. 
    Used to evaluate how well rounded answers will do. 
    """
    #ignore certain entries of M_hat (those that don't correspond to data we possess.  )
    numEntries = cp.sum(B)
    mat = M-(M_hat*B)

    lossValue = cp.square(cp.linalg.norm(mat))/numEntries
    return lossValue
def roundMatrixFringes(M_hat):
    """
    Rounds value of the matrix to be within the valid range. 
    """
    smallValue = .5
    bigValue = 5
    predTooSmall = M_hat<smallValue
    predTooBig = M_hat > bigValue
    #elements neither too big nor too small. 
    either = cp.logical_not(cp.logical_or(predTooBig, predTooSmall))
    M_hat = either*M_hat + smallValue*predTooSmall + bigValue*predTooBig
    return M_hat
def roundFringePredictions(predictions):
    """
    Rounds list of predictions to be within the valid range. 
    """
    numPredictions = predictions.shape[0]
    print("number of predictions to make: ", numPredictions)
    smallValue = .5
    bigValue = 5
    predTooSmall = predictions<smallValue
    predTooBig = predictions>bigValue
    print("num too small: ", cp.sum(predTooSmall)/numPredictions)
    print("num too big: ", cp.sum(predTooBig)/numPredictions)

    #logical and of two arrays should be 0 bc nothing in common. 
    either = cp.logical_not(cp.logical_or(predTooBig, predTooSmall))
    #multiply elements of predictions which are invalid by 0, then add small value or big values to those elements to get the new value. 
    predictions = either*predictions + predTooSmall*smallValue + predTooBig*bigValue
    return predictions

def printToFile(predictions, url):
   #predictionStrings = predictions.astype(str)
   # print(predictionStrings)
    with open(url, "w") as f:
        for i in range(0,predictions.shape[0]):
            f.write(str(predictions[i]) + "\n")
    return

def chooseR(n,m, M,B):
    r = 6
    return r
#UNUSED METHODS BELOW HERE. THEY WERE USED PREVIOUSLY BUT BECAME OBSOLETE. 
def roundMatrix(M_hat):
    """
    Rounds the values in M_hat to the nearest multiple of .5. Also rounds up to .5 and down to 5. 
    """
    smallValue = .5
    bigValue = 5
    predTooSmall = M_hat<smallValue
    predTooBig = M_hat > bigValue
    #elements neither too big nor too small. 
    either = cp.logical_not(cp.logical_or(predTooBig, predTooSmall))
    M_hat = either*M_hat + smallValue*predTooSmall + bigValue*predTooBig
    del predTooSmall
    del predTooBig
    del either

    #rounding part. 
    M_hat = 2*M_hat
    
    floor = cp.floor(M_hat)
    ceiling = cp.ceil(M_hat)

    distBelow = M_hat - floor
    distAbove = ceiling-M_hat
     
    del M_hat
    #elements which are closer to lower than higher should be rounded down. 
    roundDown = distBelow<=distAbove
    del distAbove
    del distBelow
    combined = floor*roundDown + ceiling*cp.logical_not(roundDown)
    #assert(cp.all(cp.abs(combined - M_hat) <=distAbove))
    #assert(cp.all(cp.abs(combined - M_hat) <=distBelow))
    del roundDown
    M_hat = combined/2
    del combined
    threshold = .01
    #assert(cp.all(M_hat>=.5) and cp.all(M_hat<=5) and cp.all((M_hat%.5)<threshold))
    return M_hat
def roundPredictions(predictions):
    """
    Find the closest multiple of .5 to the value, and make it that value. 

    """
    numPredictions = predictions.shape[0]
    print("number of predictions to make: ", numPredictions)
    smallValue = .5
    bigValue = 5
    predTooSmall = predictions<smallValue
    predTooBig = predictions>bigValue
    print("num too small: ", cp.sum(predTooSmall)/numPredictions)
    print("num too big: ", cp.sum(predTooBig)/numPredictions)

    #logical and of two arrays should be 0 bc nothing in common. 
    either = cp.logical_not(cp.logical_or(predTooBig, predTooSmall))
    #multiply elements of predictions which are invalid by 0, then add small value or big values to those elements to get the new value. 
    predictions = either*predictions + predTooSmall*smallValue + predTooBig*bigValue
    val = predictions*2
    floor = cp.floor(val)
    ceiling = cp.ceil(val)
    #know val>floor
    distBelow = val-floor
    #know ceiling > val
    distAbove = ceiling - val
    #boolean which is 1 if want to round down
    roundDown = distBelow<=distAbove
    combined = floor*roundDown + ceiling*(cp.logical_not(roundDown))
    assert(cp.all(cp.abs(combined - val) <=distAbove))
    assert(cp.all(cp.abs(combined - val) <=distBelow))
    normal = combined/2
    threshold = .01
    assert(cp.all(normal>=.5) and cp.all(normal<=5) and cp.all((normal%.5)<threshold))
    return normal
def fiveReg(X,Y, isItX):
    """
    Regularization to check that the values of XY^T are less than 5 and greater than 0.  Gradient value. 
    """
    l = 1
    M = X@Y.T
    timeBool = time.time()
    I = (M>5).astype(int)
    #changed from 0 to .5
    J = (M<.5).astype(int)
    timeDone = time.time()
    timeBoolTotal = timeDone - timeBool
    #print("bool time: ", timeBoolTotal)
    timeMiddle = time.time()
    #o = cp.ones(shape = M.shape)
    #fiveTerm = (M-5*o)*I
    #assert(cp.all(fiveTerm>=0))
    #zeroTerm = (M-.5*o)*J
    valTerm = M*(I+J)-(5*I + .5*J)
    #assert(cp.all(zeroTerm<=0))
    timeMiddleEnd = time.time()
    #print("time middle: ", timeMiddleEnd - timeMiddle)
    timeStart = time.time()
    
    if(isItX):
        term = (valTerm)@Y
    else:
        term = (valTerm).T @X
    timeEnd = time.time()
    #print("time final: ", timeEnd - timeStart)
    return 2*l*term.get()

def checkLoss(M, B,X,Y, loss):
    """
    A function which checks the loss function to make sure it's working properly. 
    """
def checkGradientCalculations():
    """
    Checks if the calculations  for the gradient are working, not apart of the 
    main submission. 
    """
    n,m, ratingMatrix, predictMatrix = importDataFromFile()
    M, B = getBooleanMatrixM(ratingMatrix, n, m)
    r = 10
    X = np.random.standard_normal((n, r))
    Y = np.random.standard_normal((m,r))
    Xgrad = calculateGradient(X,Y,M,B,True, True)
    Ygrad = calculateGradient(X,Y,M,B,False,  True)
    return
def stochasticGradientDescent(Mtrain, Mtest, X, Y, Btrain, Btest, learningRate, numEpochs):
    """
    Another version of the gradient descent method, but this one instead of updating the entire matrix all at once, does it row by row, 
    randomized. Thus the batches are the number of examples in each row of the matrix. 
    """
    X0 = X
    Y0 = Y
    n = X.shape[0]
    m = Y.shape[0]
    s = max(m, n)
    X1 = X0
    Y1 = Y0
    for i in range(0, numEpochs):
        perm = np.arange(0, s)
        np.random.shuffle(perm)
        startLoop = time.time()
        sumTime = 0
        for j in range(0, s):
            index = perm[j]
            #in cases where the random index is bigger than one of the arrays, it just does only the other one. 
            if(index<n):
                #x gradient for a given row (calculated same as the overall gradient)
                startGrad = time.time()
                xGrad = calculateGradientRow(X0, Y0, Mtrain, Btrain, index,True)
                #xGrad = calculateGradient(X0, Y0, Mtrain, Btrain, True)[index, :]
                
                #print("time grad: ", timeGrad)
                
                X1[index,:] = X0[index,:] - learningRate*xGrad
                X0 = X1
            if(index<m):
                yGrad = calculateGradientRow(X1, Y0, Mtrain, Btrain, index, False)
                #yGrad = calculateGradient(X1, Y0, Mtrain, Btrain, True)[index, :]
                
                Y1[index, :] = Y0[index,:] - learningRate*yGrad
                Y0 = Y1
            endGrad = time.time()
            timeGrad = endGrad - startGrad
            sumTime+=timeGrad
        avgTime = sumTime/s
        print("avg time: ", avgTime)
            #IMPLEMENT SOME WAY OF CONTINUOUS LOSS PRINTING HERE, throughout a given epoch. 
        endLoop = time.time()
        timeLoop = endLoop-startLoop
        print("loop time: ", timeLoop)
        trainLoss = loss(Mtrain, Btrain, X1, Y1)
        testLoss = loss(Mtest, Btest, X1, Y1)
        print("Epoch {epoch} train loss: {tloss}, test loss: {vloss}".format(epoch=i, tloss = trainLoss, vloss = testLoss))
    return X1, Y1
def gradientDescent(Mtrain, Mtest, X, Y, Btrain, Btest, learningRate, numEpochs):
    """
    perform GRADIENT Descent using alternating minimization on X and Y. 
    note that this calculates the gradient of the ENTIRE matrix X, using ALL the training 
    data given to us(ie the stuff in M). This is NOT stochastic gradient descent. 
    Will do that in a separate method. Thus, each eopoch is just one training step here. As you iterate 
    through the entire dataset each gradient update. 
    """
    X0 = X
    Y0= Y
    #could implement k fold cross validation here too with rows potentially
    for i in range(0, numEpochs):
        #note: could potentially alter the inputted M here to make it more robust instead. 
        X1 = X0 - learningRate*calculateGradient(X0, Y0,Mtrain,Btrain,True)
        Y1 = Y0 - learningRate*calculateGradient(X1, Y0, Mtrain, Btrain, False)
        #want these to be copies, not a reference to the original? 
      
        trainLoss = loss(Mtrain, Btrain, X1, Y1)
        testLoss = loss(Mtest, Btest, X1,Y1)
        print("Epoch {epoch} train loss: {tloss}, test loss: {vloss}".format(epoch=i, tloss = trainLoss, vloss = testLoss))
        Y0 = Y1
        X0 = X1
    return X1, Y1
def calculateGradientRow(X,Y,M,B,row, isItX,check = False):
    """
    Like calculate gradient, but only does it for a specific row. This mixes better with stochastic grad descent method.
    """
    n = X.shape[0]
    m = Y.shape[0]
    if(isItX):
        
        term = M[row,:] - Y@X[row,:] * B[row, :]
        #term = M[row, :] - cp.multiply(cp.matmul(Y, X[row, :]), B[row, :])
        assert(term.shape == (m,))
        #should be term@Y but with term as a row vector. 
        return -2*Y.T@term
    else:
        term = M[:, row] - (X@Y[row, :])*B[:, row]
        #term = M[:, row] - (cp.multiply(cp.matmul(X, Y[row, :]), B[:, row]))
        assert(term.shape == (n,))
        return -2* X.T@term

runProgram()