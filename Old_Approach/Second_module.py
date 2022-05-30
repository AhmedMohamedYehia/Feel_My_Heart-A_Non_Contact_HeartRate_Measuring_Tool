
from skimage.color import rgb2gray,rgb2hsv
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from numpy.linalg import eig, pinv
from scipy.signal import butter, lfilter, freqz


# In[2]:


def show_images(images,titles=None):
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()
    
def convertImageToRGBChannels(img):
    red_channel = img[:,:,0]
    red_img = np.zeros(img.shape)
    red_img[:,:,0] = red_channel
    
    blue_channel = img[:,:,2]
    blue_img = np.zeros(img.shape)
    blue_img[:,:,2] = blue_channel
       
    green_channel = img[:,:,1]
    green_img = np.zeros(img.shape)
    green_img[:,:,1] = green_channel   
    return red_img,green_img,blue_img
    
def extractColor(frame):
    mean = np.mean(frame[:,:,:])
    return mean 


# In[3]:


## read images and setting it in vector
def RGBimages(images):

    red_images=[]
    green_images=[]
    blue_images=[]
    for img in images:
#         img = plt.imread(str(i)+'.png')
        R,G,B=convertImageToRGBChannels(img)
        red_images.append(R[:,:,0:3])
        blue_images.append(B[:,:,0:3])
        green_images.append(G[:,:,0:3])
    return red_images,green_images,blue_images



def detrendingNormalizationPlotting(red_images,green_images,blue_images):
    mean_red_list=[]
    mean_blue_list=[]
    mean_green_list=[]
    x_list=[]
    for i in range (len(red_images)):
        x_list.append(i+1)
    ##Spatially averaging for each frame in RGB
    for red_channel in red_images:
        mean = extractColor(red_channel)
        mean_red_list.append(mean)

    for blue_channel in blue_images:
        mean = extractColor(blue_channel)
        mean_blue_list.append(mean)

    for green_channel in green_images:
        mean = extractColor(green_channel)
        mean_green_list.append(mean)
    ## detrending the produced signals
    detrended_red=signal.detrend(mean_red_list)
    detrended_green=signal.detrend(mean_green_list)
    detrended_blue=signal.detrend(mean_blue_list)
    ## Normalizing the detrended signals
    detrended_normalized_red=(detrended_red-(np.mean(detrended_red)))/(np.std(detrended_red))
    detrended_normalized_green=(detrended_green-(np.mean(detrended_green)))/(np.std(detrended_green))
    detrended_normalized_blue=(detrended_blue-(np.mean(detrended_blue)))/(np.std(detrended_blue))
        ## Plotting
    #####################################################################################
    plt.subplot(1, 2, 1)
    plt.plot(x_list,detrended_normalized_red, 'r')
    plt.xlabel('num of frames')
    plt.ylabel('amplitude')
    plt.title('Detrended Normalized Red channel')

    plt.subplot(1, 2, 2)
    plt.plot(x_list, mean_red_list)
    plt.xlabel('num of frames')
    plt.ylabel('amplitude')
    plt.title('Red channel')
    plt.tight_layout()
    plt.show()
    
    ########################################################################################
    plt.subplot(1, 2, 1)
    plt.plot(x_list, detrended_normalized_green,'g')
    plt.xlabel('num of frames')
    plt.ylabel('amplitude')
    plt.title('Detrended Normalized Green channel')

    plt.subplot(1, 2, 2)
    plt.plot(x_list, mean_green_list)
    plt.xlabel('num of frames')
    plt.ylabel('amplitude')
    plt.title('Green channel')
    plt.tight_layout()
    plt.show()
    ###################################################################################
    
    plt.subplot(1, 2, 1)
    plt.plot(x_list, detrended_normalized_blue,'b')
    plt.xlabel('num of frames')
    plt.ylabel('amplitude')
    plt.title('Detrended Normalized Blue channel')

    plt.subplot(1, 2, 2)
    plt.plot(x_list, mean_blue_list)
    plt.xlabel('num of frames')
    plt.ylabel('amplitude')
    plt.title('Blue channel')
    plt.tight_layout()
    plt.show()
    array_tuple = (detrended_normalized_red, detrended_normalized_green, detrended_normalized_blue)
    Matrix = np.vstack(array_tuple)

    return Matrix


def ROItoRGBchannels(images):
    red_images,green_images,blue_images=RGBimages(images)
    Matrix=detrendingNormalizationPlotting(red_images,green_images,blue_images)
    return Matrix
# Matrix=ROItoRGBchannels(images)


# # Second module (ICA)

# In[7]:


def whiteningData(Matrix):
    #obtaining PCA on 3 components
    Matrix = np.matrix(Matrix.astype(np.float64))
    PCA = np.asarray([2,1,0])
    m =int(Matrix.shape[0])
    T=Matrix.shape[1]
    [eigValues,eigVectors] = np.linalg.eig((Matrix * Matrix.T) / float(T))
    argSort = eigValues.argsort()
    Ds = eigValues[argSort]
    temp=np.sqrt(Ds[PCA])
    B = eigVectors[:,argSort[PCA]] 
    #Scaling of the PCA
    Diagonales=np.diag(1/temp)
    B = Diagonales * B.T 
    #B is whitening matrix
    PWT=B * Matrix 
    #PWT is whitened matrix   
    #Matrix has been transformed into a set of PCA loadings with equal variances which is the transpose of the PWT
    return PWT,B

def constructCumlant(Matrix,PWT):
    #here we make the cumulant matrix of the independent components in the ICA
    m=int(Matrix.shape[0])
    T=Matrix.shape[1]
    PW= PWT.T
    #the dimensions of cum matrix is n*n*n*n 
    #as we will get 3 ICs so the dimension will be 3*3*3*3
    CumMat = np.matrix(np.zeros([m,m*6], dtype = np.float64))
    SinMat = np.matrix(np.eye(m, dtype=np.float64))
    Temp = np.zeros([m, m], dtype =np.float64)
    Xi=Xij = np.zeros(m, dtype=np.float64)
    Range = np.asarray([0,1,2])
    #filling the cumlant matrix
    for i in range(m):
        Xi = PW[:,i]
        Xij = np.multiply(Xi, Xi)
        temp1=np.multiply(Xij, PW).T * PW / float(T)
        temp2=np.dot(SinMat[:,i], SinMat[:,i].T)
        CumMat[:,Range]=temp1-SinMat-2*temp2
        Range = Range + m
        for j in range(i):
            Xij = np.multiply(Xi, PW[:,j])
            temp1=np.multiply(Xij, PW).T * PW / float(T)
            temp2=SinMat[:,i] * SinMat[:,j].T
            temp3=SinMat[:,j] * SinMat[:,i].T
            CumMat[:,Range]= np.sqrt(2) *temp1-temp2-temp3
            Range = Range + m
    return CumMat

def JointDiagonalization(Matrix,CumMat):
    #aims at minimizing the sum-of-squares of the offdiagonal elements 
    m =int(Matrix.shape[0])
    T=Matrix.shape[1]
    Diag = np.zeros(3, dtype=np.float64)
    Range = np.asarray([0,1,2])
    V=np.matrix(np.eye(m, dtype=np.float64))
    g = np.zeros([2,6], dtype=np.float64) 
    G = np.zeros([2,2], dtype=np.float64) 
    #rotation matrix
    thres = 1.0e-6 / np.sqrt(T)
    boolen = True
    c=s=Ondiag=Offdiag=theta=0
    while boolen:
        boolen = False
        for i in range(m-1):
            for j in range(i+1, m): 
                I1 = np.arange(i, m*6, m) 
                I2 = np.arange(j, m*6, m) 
                #computation of angle
                Sub1=CumMat[i, I1] - CumMat[j, I2]
                Sub2=CumMat[i, I2] + CumMat[j, I1]
                g = np.concatenate([Sub1, Sub2])
                Ondiag = np.diag(np.matmul(g, g.T))[0]-np.diag(np.matmul(g, g.T))[1]
                Offdiag = np.matmul(g, g.T)[0, 1] + np.matmul(g, g.T)[1, 0] 
                Sqroot= np.sqrt(Ondiag * Ondiag + Offdiag * Offdiag)
                theta = 0.5 * np.arctan2(Offdiag, Ondiag +Sqroot) 
                if abs(theta) > thres:
                    boolen = True
                    cos = np.cos(theta)
                    sin = np.sin(theta)
                    G = np.matrix([[cos, -sin] , [sin, cos] ]) 
                    index = np.array([i, j])
                    V[:,index] = V[:,index] * G
                    #multiply element wise
                    CumMat[index,:] = G.T * CumMat[index,:]
                    concat=np.concatenate([I1, I2])
                    temp1=cos*CumMat[:,I1]+sin*CumMat[:,I2]
                    temp2=-sin*CumMat[:,I1]+cos*CumMat[:,I2]
                    CumMat[:, concat] = np.append( temp1, temp2, axis=1)
    return V

def arranging_norming(B,V):
    #arranging the rows of the demixing matrix according to the normof columns of the inverse of the demixing matrix
    #in order to get the most energetic components first
    B = V.T * B #Demixing matrix
    A = np.linalg.inv(B) 
    b = B[:,0]
    #deal with signs speacially when the number is 0 not positive or negative
    mat=np.zeros(3,dtype=np.float64)
    for i in range (3):
        if np.sign(b[i])>=0:
            mat[i]=1
        else:
            mat[i]=-1
    B = np.diag(np.array(mat)) * B 
    return B


# In[8]:


def plotICA(Matrix,Sources):
    x_list=[]
    for i in range (len(Sources[0])):
        x_list.append(i+1)
    plt.plot(x_list, Sources[0])
    plt.xlabel('num of frames')
    plt.ylabel('brightness')
    plt.title('first independent source')
    plt.tight_layout()
    plt.show()

    plt.plot(x_list, Sources[1])
    plt.xlabel('num of frames')
    plt.ylabel('brightness')
    plt.title('second independent source')
    plt.tight_layout()
    plt.show()
    plt.plot(x_list, Sources[2])
    plt.xlabel('num of frames')
    plt.ylabel('brightness')
    plt.title('third independent source')
    plt.tight_layout()
    plt.show()



def ICA(Matrix):
    PWT,B=whiteningData(Matrix)
    CM=constructCumlant(Matrix,PWT)
    V=JointDiagonalization(Matrix,CM)
    B=arranging_norming(B,V)
    Sources=np.asarray(B*np.matrix(Matrix))
    return Sources


def filters ( data, lowcut, highcut, fs, order,Type):
    if Type=='low':
        nyq = 0.5 * fs
        normal_cutoff = lowcut / nyq
        b, a = signal.butter(order, normal_cutoff, btype=Type, analog=False)
        return signal.lfilter(b, a, data)
    elif Type=='band':
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype=Type)
        return signal.lfilter(b, a, data)
        


def lowPassFilter(Sources):
    lowPass0 = filters(Sources[0], 5,0, 50, 1,'low')
    lowPass1 = filters(Sources[1], 5,0, 50, 1,'low')
    lowPass2 = filters(Sources[2], 5,0, 50, 1,'low')
    return lowPass0,lowPass1,lowPass2        

def bandPassFilter(lowPass0,lowPass1,lowPass2):
    bandPass0=filters(data=lowPass0,lowcut=0.75,highcut=3,fs=55,order=2,Type='band')
    bandPass1=filters(data=lowPass1,lowcut=0.75,highcut=3,fs=55,order=2,Type='band')
    bandPass2=filters(data=lowPass2,lowcut=0.75,highcut=3,fs=55,order=2,Type='band')
    return bandPass0,bandPass1,bandPass2



def filtering(Sources):
    l0,l1,l2=lowPassFilter(Sources)
    b0,b1,b2=bandPassFilter(l0,l1,l2)
    return b0,b1,b2

