import numpy as np
from .gendata import generate
from . import OptimalTransport
from . import ForwardSelection as FS
from . import BackwardSelection as BS
from . import overconditioning 
from . import parametric
from scipy.linalg import block_diag
from mpmath import mp
mp.dps = 500


def compute_p_value(intervals, etaT_Y, etaT_Sigma_eta):
    denominator = 0
    numerator = 0

    for i in intervals:
        leftside, rightside = i
        if leftside <= etaT_Y <= rightside:
            numerator = denominator + mp.ncdf(etaT_Y / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))
        denominator += mp.ncdf(rightside / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))
    cdf = float(numerator / denominator)
    return 2 * min(cdf, 1 - cdf)


def SI_SeqFS_DA(Xs, Ys, Xt, Yt, k, Sigma_s, Sigma_t, zmin=-20, zmax=20, method = 'forward', jth = None):
    if method not in ['forward', 'backward']:
        raise ValueError('method must be either \'forward\' or \'backward\'')
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

    Xt_M = Xt[:, sorted(SELECTION_F)].copy()
    # print('Selected features:', SELECTION_F)
    if jth == None:
        jth = np.random.choice(range(len(SELECTION_F)))
    if jth >= len(SELECTION_F):
        raise ValueError('jth must be less than the number of features selected')
    # print(f"{jth}-th feature in {SELECTION_F}")
    ej = np.zeros((len(SELECTION_F), 1))
    ej[jth][0] = 1

    Zeta = np.concatenate((np.zeros((nt, ns)), np.identity(nt)), axis = 1)
    
    eta = np.vstack((np.zeros((ns, 1)), Xt_M.dot(np.linalg.inv(Xt_M.T.dot(Xt_M))).dot(ej)))
    eta = eta.reshape((-1,1))
    etaT_Sigma_eta = np.dot(np.dot(eta.T , Sigma) , eta).item()
    
    I_nplusm = np.identity(ns+nt)
    b = np.dot(Sigma, eta) / etaT_Sigma_eta
    a = np.dot((I_nplusm - np.dot(b, eta.T)), Y)

    etaTY = np.dot(eta.T, Y).item()
    if method == 'forward':
        if type(k) == int:
            finalinterval = parametric.para_DA_FSwithfixedK(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F, zmin, zmax)
        else:
            finalinterval = parametric.para_DA_FSwithStoppingCriterion(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F, k, zmin, zmax)
    elif method == 'backward':
        if type(k) == int:
            finalinterval = parametric.para_DA_BS(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F, zmin, zmax)  
        else:
            finalinterval = parametric.para_DA_BSwith_stoppingCriteria(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F,typeCrit=k, z_min=zmin, z_max= zmax)

    selective_p_value = compute_p_value(finalinterval, etaTY, etaT_Sigma_eta)
    return selective_p_value
    

if __name__ == "__main__":
    ns = 50 #number of source's samples
    nt = 10 #number of target's samples
    p = 4 #number of features
    
    true_beta_s = np.full((p,1), 2) #source's beta
    true_beta_t = np.full((p,1), 0) #target's beta

    Xs, Xt, Ys, Yt, Sigma_s, Sigma_t = generate(ns, nt, p, true_beta_s, true_beta_t)

    K = 3 # number of features to be selected
    print(SI_SeqFS_DA(Xs, Ys, Xt, Yt, K, Sigma_s, Sigma_t, method='forward', jth=None)) #jth = None means randomly choose jth

    K = 'AIC' # stopping criterion
    print(SI_SeqFS_DA(Xs, Ys, Xt, Yt, K, Sigma_s, Sigma_t, method='backward', jth=1))