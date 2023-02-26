import numpy as np 
import cupy as cp
from HW1helper import *
#from HW1helper import importDataFromFile
"""
Want to train on completing the matrix, ie M = XY^T. 
Loss function is sum of squared error but only on the examples we have in training data. 
want to hold out some data as training data. 

"""
def runProgram():
    small = "mat_comp_small"
    big = "mat_comp"
    file_name = big
    n,m, ratingMatrix, predictMatrix = importDataFromFile(file_name)
    (Mtrain, Btrain), (Mval,Bval), (Mtest,Btest) = readyData(n,m,ratingMatrix)
    r = chooseR(n, m, Mtrain, Btrain)
    X,Y = initialize(n,m, r)
    numEpochs = 500
    learningRate = .00004
    batchSize = 10
    #Xfinal, Yfinal = stochasticGradientDescent(Mtrain, Mval, X, Y, Btrain, Bval, learningRate, numEpochs)
    Xfinal, Yfinal = batchGradientDescent(Mtrain, Mval, X, Y, Btrain, Bval, learningRate, numEpochs, batchSize)
    testLoss = loss(Mtest, Btest, Xfinal, Yfinal)
    print("test loss is: ", testLoss)
    predictions = predict(Xfinal, Yfinal, predictMatrix)
    print(predictions)
    return
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
def batchGradientDescent(Mtrain, Mtest, X, Y, Btrain, Btest, learningRate, numEpochs, batchSize):
    """
    Calculates a gradient on batchSize adjacent rows in X or Y. This is much faster than the row based one, but 
    potentially the gradients are less exact. 
    """
    X0 = X
    Y0 = Y
    n = X.shape[0]
    m = Y.shape[0]
    s = max(m, n)
    #difference in size between the two
    
    X1 = X0
    Y1 = Y0
    #assuming int rounds down. 
    xBatches = int(n/batchSize) + 1
    yBatches = int(m/batchSize)+ 1
    #one group has leftovers, other has leftovers+d. know leftovers<bs. 
    for i in range(0, numEpochs):
        biggestBatchSize = max(xBatches, yBatches)
        perm = np.arange(0, biggestBatchSize)
        np.random.shuffle(perm)
        startLoop = time.time()
        for b in perm:
            MbatchX = Mtrain[batchSize*b:batchSize*(b+1),:]
            BbatchX = Btrain[batchSize*b:batchSize*(b+1), :]
            MbatchY = Mtrain[:, batchSize*b:batchSize*(b+1)]
            BbatchY = Btrain[:, batchSize*b:batchSize*(b+1)]

            if(b<xBatches):
                Xbatch = X0[batchSize*b:batchSize*(b+1),:]
                
                xGrad = calculateGradient(Xbatch, Y, MbatchX, BbatchX, True)
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
def loss(M, B, X, Y):
    """
    The base loss function is: 
    f(X,Y) = sum_{i,j in omega}(Mij - (XY^T)ij)^2
    We can rwerite this in matrix form however, using B. 
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
    startLoss = time.time()
    #cast to cp to make this operation faster. 
    numEntries = cp.sum(cp.asarray(B))
    mat = cp.asarray(M)-(cp.asarray(X)@cp.asarray(Y).T)*cp.asarray(B)
    #mat = M-np.multiply((cp.matmul(X, cp.transpose(Y))), B)
    #squared frobenius norm. 
    #normal frobenius norm:
    #this method is WAY faster. However, need to square it somehow. 
    #still faster with these extra things added. 
    loss = cp.square(cp.linalg.norm(mat))/numEntries
    endLoss = time.time()
    
    print("lsos time: ", endLoss-startLoss)
    return loss

def checkLoss(M, B,X,Y, loss):
    """
    A function which checks the loss function to make sure it's working properly. 
    """
def predict(X, Y, predictData):
    """
    Calculate predictions for the data. 
    Predictions: q x 2, where first column is user second column is movie. Indexed starting at one. 
    To predict, just pick the ERM, ie the value of XY^T at i,j
    """
    Mfinal = X@Y.T
    predictions = Mfinal[predictData[:, 0]-1, predictData[:, 1]-1]
    assert(checkPredictions(predictions, Mfinal, predictData))

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
    """
    
    term = (M-(X@Y.T)*B)
    if(isItX):
        grad = term@Y
        assert(grad.shape == X.shape)
    else:
        grad = term.T @X
        assert(grad.shape == Y.shape)
    if(check):
        assert(checkGradient(X,Y,M,B, isItX, grad))
    return -2*grad
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
def chooseR(n,m, M,B):
    r = 6
    return r

runProgram()