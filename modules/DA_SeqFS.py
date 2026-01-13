import numpy as np
from . import ForwardSelection as FS
from . import OptimalTransport
from scipy.linalg import block_diag
from . import BackwardSelection as BS

def DA_SeqFS(Xs, Ys, Xt, Yt, Sigma_s, Sigma_t, k, method):
    ns = Xs.shape[0]
    nt = Xt.shape[0]
    p = Xs.shape[1]
    
    Xs_ = np.concatenate((Xs, Ys), axis = 1)
    Xt_ = np.concatenate((Xt, Yt), axis = 1)
    XsXt_ = np.concatenate((Xs_, Xt_), axis= 0)

    X = np.concatenate((Xs, Xt), axis= 0)
    Y = np.concatenate((Ys, Yt), axis= 0)

    Sigma = block_diag(Sigma_s, Sigma_t)
    h = np.concatenate((np.ones((ns, 1))/ns, np.ones((nt,1))/nt), axis = 0) 
    S = OptimalTransport.convert(ns,nt)
    # remove last row
    S_ = S[:-1].copy()
    h_ = h[:-1].copy()
    # Gamma drives source data to target data 
    GAMMA = OptimalTransport.solveOT(ns, nt, S_, h_, XsXt_)['gamma']

    Xtilde = np.dot(GAMMA, X)
    Ytilde = np.dot(GAMMA, Y)
    Sigmatilde = GAMMA.T.dot(Sigma.dot(GAMMA))
    if method == 'forward':
        if k == 'AIC':
            SELECTION_F = FS.SelectionAIC(Ytilde, Xtilde, Sigmatilde)
        elif k == 'BIC':
            SELECTION_F = FS.SelectionBIC(Ytilde, Xtilde, Sigmatilde)
        elif k == 'Adjusted R2':
            SELECTION_F = FS.SelectionAdjR2(Ytilde, Xtilde)
        else:
            SELECTION_F = FS.fixedSelection(Ytilde, Xtilde, k)[0]
    elif method == 'backward':
        if k == 'AIC':
            SELECTION_F = BS.SelectionAICforBS(Ytilde, Xtilde, Sigmatilde)
        elif k == 'BIC':
            SELECTION_F = BS.SelectionBIC(Ytilde, Xtilde, Sigmatilde)
        elif k == 'Adjusted R2':
            SELECTION_F = BS.SelectionAdjR2(Ytilde, Xtilde)
        else:
            SELECTION_F = BS.fixedBS(Ytilde, Xtilde, k)[0]

    return sorted(SELECTION_F)