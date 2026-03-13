import numpy as np
# from .gendata import generate
from . import OptimalTransport
from . import ForwardSelection as FS
from . import BackwardSelection as BS
from . import overconditioning 
from . import parametric
from . import parametric_old 
# from . import parametric_parallel as parametric
from . import gendata
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


def SI_SeqFS_DA(Xs, Ys, Xt, Yt, k, Sigma_s, Sigma_t, zmin=-20, zmax=20, method='forward', jth=None):
    # 1. Kiểm tra tính hợp lệ của method
    method_upper = 'FS' if method == 'forward' else 'BS'
    if method not in ['forward', 'backward']:
        raise ValueError("method must be either 'forward' or 'backward'")
    
    # 2. Chuẩn bị dữ liệu và tham số hệ thống
    ns, nt = Xs.shape[0], Xt.shape[0]
    p = Xs.shape[1]
    
    XsXt_ = np.concatenate((np.concatenate((Xs, Ys), axis=1), 
                            np.concatenate((Xt, Yt), axis=1)), axis=0)
    X = np.concatenate((Xs, Xt), axis=0)
    Y = np.concatenate((Ys, Yt), axis=0)
    
    Sigma = block_diag(Sigma_s, Sigma_t)
    h_ = (np.ones((ns + nt, 1)) / (ns if ns == nt else 1))[:-1].copy() # Giả định logic h_ từ code cũ
    # Lưu ý: h_ nên được tính chính xác theo trọng số ns, nt như code cũ của bạn:
    h = np.concatenate((np.ones((ns, 1))/ns, np.ones((nt, 1))/nt), axis=0)
    h_ = h[:-1].copy()
    
    S_ = OptimalTransport.convert(ns, nt)[:-1].copy()

    # 3. Giải OT ban đầu để xác định SELECTION_F (Model gốc)
    GAMMA = OptimalTransport.solveOT(ns, nt, S_, h_, XsXt_)['gamma']
    Xtilde = np.dot(GAMMA, X)
    Ytilde = np.dot(GAMMA, Y)
    Sigmatilde = GAMMA.T.dot(Sigma.dot(GAMMA))

    # Xác định k_type và criterion
    if isinstance(k, int):
        k_type = 'fixed'
        criterion = 'AIC' # Mặc định, sẽ không dùng đến khi là 'fixed'
        if method == 'forward':
            SELECTION_F = FS.fixedSelection(Ytilde, Xtilde, k)[0]
        else:
            SELECTION_F = BS.fixedBS(Ytilde, Xtilde, k)[0]
    else:
        k_type = 'stopping'
        criterion = k
        if method == 'forward':
            if k == 'AIC': SELECTION_F = FS.SelectionAIC(Ytilde, Xtilde, Sigmatilde)
            elif k == 'BIC': SELECTION_F = FS.SelectionBIC(Ytilde, Xtilde, Sigmatilde)
            elif k == 'Adjusted R2': SELECTION_F = FS.SelectionAdjR2(Ytilde, Xtilde)
        else:
            if k == 'AIC': SELECTION_F = BS.SelectionAICforBS(Ytilde, Xtilde, Sigmatilde)
            elif k == 'BIC': SELECTION_F = BS.SelectionBIC(Ytilde, Xtilde, Sigmatilde)
            elif k == 'Adjusted R2': SELECTION_F = BS.SelectionAdjR2(Ytilde, Xtilde)

    # 4. Tính toán Vector Phản chiếu (eta) để chuyển sang bài toán parametric
    if jth is None:
        jth = np.random.choice(range(len(SELECTION_F)))
    
    Xt_M = Xt[:, sorted(SELECTION_F)].copy()
    ej = np.zeros((len(SELECTION_F), 1))
    ej[jth][0] = 1
    
    # eta được thiết kế để cô lập ảnh hưởng của biến thứ j trong tập đã chọn
    eta = np.vstack((np.zeros((ns, 1)), Xt_M.dot(np.linalg.inv(Xt_M.T.dot(Xt_M))).dot(ej)))
    etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta).item()
    
    # Tính a và b cho Y(z) = a + bz
    b = np.dot(Sigma, eta) / etaT_Sigma_eta
    a = np.dot((np.identity(ns + nt) - np.dot(b, eta.T)), Y)
    etaTY = np.dot(eta.T, Y).item()

    # 5. Gọi hàm gộp duy nhất thay vì 4 hàm riêng lẻ
    import time
    t1 = time.time()
    # finalinterval = parametric_old.parametric(
    #     ns=ns, nt=nt, a=a, b=b, X=X, Sigma=Sigma, S_=S_, h_=h_, 
    #     SELECTION_F=SELECTION_F, 
    #     method=method_upper, 
    #     k_type=k_type, 
    #     criterion=criterion, 
    #     z_min=zmin, z_max=zmax
    # )

    # selective_p_value = compute_p_value(finalinterval, etaTY, etaT_Sigma_eta)
    # print(" pvalue of old parametric:", selective_p_value)
    # with open(f"exp/time_old_{ns}.txt", "a") as f:
    #     f.write(f"{time.time() - t1}\n")


    t2 = time.time()
    finalinterval = parametric.parametric(
        ns=ns, nt=nt, a=a, b=b, X=X, Sigma=Sigma, S_=S_, h_=h_, 
        SELECTION_F=SELECTION_F, 
        method=method_upper, 
        k_type=k_type, 
        criterion=criterion, 
        z_min=zmin, z_max=zmax
    )

    # 6. Tính p-value dựa trên tập các khoảng hội tụ (Truncated Normal)
    selective_p_value = compute_p_value(finalinterval, etaTY, etaT_Sigma_eta)
    print(" pvalue of new parametric:", selective_p_value)
    with open(f"exp/time_new_nt{nt}.txt", "a") as f:
        f.write(f"{time.time() - t2}\n")
    return selective_p_value

if __name__ == "__main__":
    ns = 50 #number of source's samples
    nt = 10 #number of target's samples
    p = 4 #number of features
    K = 2 #number of features to be selected
    true_beta_s = np.full((p,1), 2) #source's beta
    true_beta_t = np.full((p,1), 0) #target's beta

    Xs, Xt, Ys, Yt, Sigma_s, Sigma_t = gendata.generate(ns, nt, p, true_beta_s, true_beta_t)

    # K = 'AIC' # number of features to be selected
    print(SI_SeqFS_DA(Xs, Ys, Xt, Yt, K, Sigma_s, Sigma_t, method='forward', jth=None)) #jth = None means randomly choose jth

    K = 'AIC' # stopping criterion
    print(SI_SeqFS_DA(Xs, Ys, Xt, Yt, K, Sigma_s, Sigma_t, method='backward', jth=1))