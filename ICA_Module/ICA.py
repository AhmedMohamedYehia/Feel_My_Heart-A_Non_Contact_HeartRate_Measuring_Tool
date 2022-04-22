from numpy import abs, append, arange, arctan2, argsort, array, concatenate, \
	cos, diag, dot, eye, float32, float64, loadtxt, matrix, multiply, ndarray, \
	newaxis, savetxt, sign, sin, sqrt, zeros

from numpy.linalg import eig, pinv


def whiteningData(Matrix):
    Matrix = matrix(Matrix.astype(float64)) 
    n=Matrix.shape[0]
    T=Matrix.shape[1]
    m =int(n)
    #[D,U]
    [eigValues,eigVectors] = eig((Matrix * Matrix.T) / float(T))
    argSort = eigValues.argsort()
    Ds = eigValues[argSort]
    PCs = arange(n-1, n-m-1, -1)
    B = eigVectors[:,argSort[PCs]].T # B obtain PCA on 3 components 
    #Scaling of the principal components
    B = diag(1./sqrt(Ds[PCs])) * B #whitening matrix
    #Sphering
    PWT=B*Matrix #whitened matrix
    return PWT,B

def constructCumlant(PWT):
    n=Matrix.shape[0]
    T=Matrix.shape[1]
    m =int(n)
    PW= PWT.T
    CM = matrix(zeros([m,m*6], dtype = float64))
    SM = matrix(eye(m, dtype=float64))
    Temp = zeros([m, m], dtype = float64)
    Xi = zeros(m, dtype=float64)
    Xij = zeros(m, dtype=float64)
    Range = arange(m)
    for i in range(m):
        Xi = PW[:,i]
        Xij = multiply(Xi, Xi)
        CM[:,Range] = multiply(Xij, PW).T * PW / float(T) - SM - 2 * dot(SM[:,i], SM[:,i].T)
        Range = Range + m
        for j in range(i):
            Xij = multiply(Xi, PW[:,j])
            CM[:,Range] = sqrt(2) * multiply(Xij, PW).T * PW / float(T) - SM[:,i] * SM[:,j].T - SM[:,j] * SM[:,i].T
            Range = Range + m
    return CM

def JointDiagonalization(CM):
    n=Matrix.shape[0]
    T=Matrix.shape[1]
    m =int(n)
    Diag = zeros(3, dtype=float64)
    On = 0.0
    Range = arange(0,1,2) 
    for im in range(6): 
        Diag = diag(CM[:,Range])
        On = On + (Diag * Diag).sum(axis=0)
        Range = Range + m
    Off = (multiply(CM,CM).sum(axis=0)).sum(axis=0) - On
    seuil = 1.0e-6 / sqrt(T)
    encore = True
    sweep = 0
    updates = 0
    upds = 0
    V=matrix(eye(m, dtype=float64))
    g = zeros([2,6], dtype=float64) 
    gg = zeros([2,2], dtype=float64)
    G = zeros([2,2], dtype=float64)
    c = 0
    s = 0
    ton = 0
    toff = 0
    theta = 0
    Gain = 0
    while encore:
        encore = False
        sweep = sweep + 1
        upds = 0
        Vkeep = V

        for p in range(m-1): #m == 3
            for q in range(p+1, m): #p == 1 | range(p+1, m) == [2]

                Ip = arange(p, m*6, m) 
                Iq = arange(q, m*6, m) 

                #computation of Givens angle
                g = concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
                gg = dot(g, g.T)
                ton = gg[0,0] - gg[1,1] 
                toff = gg[0, 1] + gg[1, 0] 
                theta = 0.5 * arctan2(toff, ton + sqrt(ton * ton + toff * toff)) 
                Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0 

                if abs(theta) > seuil:
                    encore = True
                    upds = upds + 1
                    c = cos(theta)
                    s = sin(theta)
                    G = matrix([[c, -s] , [s, c] ]) 
                    pair = array([p, q])
                    V[:,pair] = V[:,pair] * G
                    CM[pair,:] = G.T * CM[pair,:]
                    CM[:, concatenate([Ip, Iq])] = append( c*CM[:,Ip]+s*CM[:,Iq], -s*CM[:,Ip]+c*CM[:,Iq], axis=1)
                    On = On + Gain
                    Off = Off - Gain

        updates = updates + upds 
    return V

def arranging_norming(B,V):
    B = V.T * B 
    A = pinv(B) 
    keys = array(argsort(multiply(A,A).sum(axis=0)[0]))[0] #[2 1 0]

    B = B[keys,:] 
    B = B[::-1,:] 
    b = B[:,0] 
    signs = array(sign(sign(b)+0.1).T)[0] #[1. -1. 1.]
    B = diag(signs) * B 
    return B