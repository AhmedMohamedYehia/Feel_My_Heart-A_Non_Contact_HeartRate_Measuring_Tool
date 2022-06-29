from hashlib import algorithms_available
import numbers
import numpy as np
import cv2
from skimage.feature import corner_harris,peak_local_max
from numpy.linalg import inv
from skimage import transform

WINDOW_SIZE = 25
HALF_WINDOW = WINDOW_SIZE // 2
ITERATIONS = 15
THERSHOLD = 1
MINIMUM_POINTS = 4


def interpolate2D(frame, meshX, meshY):

    floorMeshX = np.floor(meshX).astype(int)
    ceilMeshX = np.ceil(meshX).astype(int)
    floorMeshY = np.floor(meshY).astype(int)
    ceilMeshY = np.ceil(meshY).astype(int)

    floorMeshX = np.where(floorMeshX<0, 0, floorMeshX)
    floorMeshX = np.where(floorMeshX>=frame.shape[1]-1, frame.shape[1]-1, floorMeshX)
    ceilMeshX = np.where(ceilMeshX<0, 0, ceilMeshX)
    ceilMeshX = np.where(ceilMeshX>=frame.shape[1]-1, frame.shape[1]-1, ceilMeshX)
    floorMeshY = np.where(floorMeshY<0, 0, floorMeshY)
    floorMeshY = np.where(floorMeshY>=frame.shape[0]-1, frame.shape[0]-1, floorMeshY)
    ceilMeshY = np.where(ceilMeshY<0, 0, ceilMeshY)
    ceilMeshY = np.where(ceilMeshY>=frame.shape[0]-1, frame.shape[0]-1, ceilMeshY)

    weight1 = (1 - (meshY - floorMeshY)) * (1 - (meshX - floorMeshX))
    weight2 = (1 - (meshY - floorMeshY)) * (meshX - floorMeshX)
    weight3 = (meshY - floorMeshY) * (1 - (meshX - floorMeshX))
    weight4 = (meshY - floorMeshY) * (meshX - floorMeshX)

    return (weight1 * frame[floorMeshY, floorMeshX] + weight2 * frame[floorMeshY, ceilMeshX] + weight3 * frame[ceilMeshY, floorMeshX] + weight4 * frame[ceilMeshY, ceilMeshX]).reshape(1,-1)


def getHarrisCorners(img,bbox):
    
    bbox = bbox.astype(int)
    ROI = img[bbox[0][1]:bbox[3][1],bbox[0][0]:bbox[3][0]]
    corners = corner_harris(ROI)
    features = peak_local_max(corners,num_peaks=20,exclude_border=2)
    features[:,1] += bbox[0][0]
    features[:,0] += bbox[0][1]

    return features[:,1],features[:,0]

def fixFeatureTranslation(X,Y,meshX,meshY,img2,Ix,Iy,A,I1):
    
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    for i in range(ITERATIONS):
        meshX2 = meshX.flatten() - HALF_WINDOW
        meshX2 = meshX2 + X
        meshY2 = meshY.flatten() - HALF_WINDOW
        meshY2 = meshY2 + Y
        I2 = interpolate2D(img2Gray, meshX2, meshY2)
        It=(I2-I1).reshape((1,-1))
        IxIt = np.sum(Ix*It)
        IyIt = np.sum(Iy*It)
        b = -1 * np.array([[IxIt],[IyIt]])
        UV = np.dot(inv(A),b)
        X += UV[0,0]
        Y += UV[1,0]
        
    return X, Y

def featureTranslation(X,Y,Ix,Iy,img1,img2):
    
    meshX,meshY=np.meshgrid(np.linspace(0, 24, num=WINDOW_SIZE).astype(int),np.linspace(0, 24, num=WINDOW_SIZE).astype(int))
    img1Gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    meshX1 = meshX.flatten() - HALF_WINDOW
    meshX1 = meshX1 + X
    meshY1 = meshY.flatten() - HALF_WINDOW
    meshY1 = meshY1 + Y
    Ix = interpolate2D(Ix, meshX1, meshY1)
    Iy = interpolate2D(Iy, meshX1, meshY1)
    IxIx = np.sum(Ix*Ix)
    IxIy = np.sum(Ix*Iy)
    IyIy = np.sum(Iy*Iy)
    A = np.array([[IxIx,IxIy],[IxIy,IyIy]])
    I1 = interpolate2D(img1Gray, meshX1, meshY1)
    
    return fixFeatureTranslation(X,Y,meshX,meshY,img2,Ix,Iy,A,I1)

def featuresTranslation(oldXPos,oldYPos,img1,img2):
    
    I = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    Iy, Ix = np.gradient(I.astype(float))
    newXPos = np.zeros(oldXPos.shape)
    newYPos = np.zeros(oldYPos.shape)
    for i in range(oldXPos.shape[0]):
        newXPos[i], newYPos[i] = featureTranslation(oldXPos[i], oldYPos[i], Ix, Iy, img1, img2)
        
    return newXPos, newYPos

def applyGeometricTransformation(oldXPos, oldYPos, newXPos, newYPos, bbox):
    oldFrame = np.zeros((oldXPos.shape[0],2))
    oldFrame[:,0] = oldXPos
    oldFrame[:,1] = oldYPos
    newFrame = np.zeros((newXPos.shape[0],2))
    newFrame[:,0] = newXPos
    newFrame[:,1] = newYPos
    trans = transform.SimilarityTransform()
    trans.estimate(dst=newFrame, src=oldFrame)
    mat = trans.params
    ones = np.ones([1,np.shape(oldFrame)[0]])
    tempMat = np.vstack((oldFrame.T.astype(float),ones))
    projectedPoints = mat.dot(tempMat)
    distance = np.square(projectedPoints[0:2,:].T - newFrame).sum(axis = 1)
    if oldFrame[distance < THERSHOLD].shape[0] < MINIMUM_POINTS:
        newInliers = newFrame
        oldInliers = oldFrame
    else:
        newInliers = newFrame[distance < THERSHOLD]
        oldInliers = oldFrame[distance < THERSHOLD]
    trans.estimate(dst=newInliers, src=oldInliers)
    mat = trans.params
    oldBboxMat = np.vstack((bbox.T,np.ones(4,dtype=int)))
    newBboxMat = mat.dot(oldBboxMat)
    newBbox = newBboxMat[0:2,:].T
    newXPos = newXPos[distance < THERSHOLD]
    newYPos = newYPos[distance < THERSHOLD]

    return newXPos, newYPos, newBbox