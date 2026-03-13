from . import overconditioning
from . import overconditioningBS
import numpy as np
from . import OptimalTransport
from . import BackwardSelection 
from . import ForwardSelection
from . import intersection
def parametric(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F, 
                            method='FS', k_type='fixed', criterion='AIC', 
                            z_min=-20, z_max=20, seed=0):
    """
    Hàm tổng quát cho Parametric Domain Adaptation với Forward/Backward Selection.
    
    Parameters:
    - method: 'FS' (Forward Selection) hoặc 'BS' (Backward Selection)
    - k_type: 'fixed' (Số lượng biến cố định) hoặc 'stopping' (Dựa trên tiêu chí dừng)
    - criterion: 'AIC', 'BIC', hoặc 'Adjusted R2'
    """
    valid_methods = ['FS', 'BS']
    valid_k_types = ['fixed', 'stopping']
    valid_criteria = ['AIC', 'BIC', 'Adjusted R2']

    if method not in valid_methods:
        raise ValueError(f"Invalid method: '{method}'. Expected one of {valid_methods}")
    
    if k_type not in valid_k_types:
        raise ValueError(f"Invalid k_type: '{k_type}'. Expected one of {valid_k_types}")
    
    # Chỉ kiểm tra criterion nếu dùng k_type là 'stopping'
    if k_type == 'stopping' and criterion not in valid_criteria:
        raise ValueError(f"Invalid criterion: '{criterion}' for stopping rule. Expected one of {valid_criteria}")
    
    # Kiểm tra tính logic của dữ liệu đầu vào
    if z_min >= z_max:
        raise ValueError(f"z_min ({z_min}) must be less than z_max ({z_max})")
    # ----------------------------------------------
    TD = []
    z = z_min
    zmax = z_max
    list_DAitv = []
    while z < zmax:
        z += 0.00001
        
        # F(z) -> intervalinloop = [left, right] -------------
        # 1. Cập nhật dữ liệu theo tham số z
        Ydeltaz = a + b * z
        XsXt_deltaz = np.concatenate((X, Ydeltaz), axis=1).copy()
        
        # 2. Giải Optimal Transport
        GAMMAdeltaz, basis_var_deltaz = OptimalTransport.solveOT(ns, nt, S_, h_, XsXt_deltaz).values()
        OMEGA = OptimalTransport.constructOMEGA(ns, nt)
        c_ = np.zeros((ns * nt, 1))
        for i in range(X.shape[1]):
            c_ += (OMEGA.dot(X[:, [i]])) * (OMEGA.dot(X[:, [i]]))

        itvDA = overconditioning.interval_DA(ns, nt, c_, basis_var_deltaz, S_, h_, a, b)
        for left, right in itvDA:
            if left <= z <= right:
                list_DAitv.append([left, right, basis_var_deltaz])
                break
        z = itvDA[-1][1]


    for ml, mr, B in list_DAitv:
        # print(f"Processing interval [{ml}, {mr}], Basis: {len(B)}, {[int(x) for x in B]}")
        z = ml
        while z < mr:
            Ydeltaz = a + b * z
            T = np.zeros(ns * nt)

            x_B = np.linalg.inv(S_[:, B]).dot(h_)
            T[B] = x_B.flatten()
            T =  T.reshape((ns, nt))

            GAMMAdeltaz = OptimalTransport.constructGamma(ns, nt,T)
            Xtildeinloop = np.dot(GAMMAdeltaz, X)
            Ytildeinloop = np.dot(GAMMAdeltaz, Ydeltaz)
            Sigmatilde_deltaz = GAMMAdeltaz.T.dot(Sigma.dot(GAMMAdeltaz))
            
            itvDA = [(ml, mr)]
            # print("DAitv_n:", itvDA)
            
            # 3. Lựa chọn Model (Selection) và Tính toán Interval
            if method == 'FS':
                if k_type == 'stopping':
                    # Tương đương para_DA_FSwithStoppingCriterion
                    if criterion == 'AIC':
                        SELECTIONinloop = ForwardSelection.SelectionAIC(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
                    elif criterion == 'BIC':
                        SELECTIONinloop = ForwardSelection.SelectionBIC(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
                    elif criterion == 'Adjusted R2':
                        SELECTIONinloop = ForwardSelection.SelectionAdjR2(Ytildeinloop, Xtildeinloop)
                    
                    intervalinloop = overconditioning.OC_Crit_interval_(
                        ns, nt, a, b, Xtildeinloop, Ytildeinloop, 
                        Sigmatilde_deltaz, SELECTIONinloop, GAMMAdeltaz, criterion
                    )
                else:
                    # Tương đương para_DA_FSwithfixedK
                    SELECTIONinloop = ForwardSelection.fixedSelection(Ytildeinloop, Xtildeinloop, len(SELECTION_F))[0]
                    intervalinloop, _, _ = overconditioning.OC_fixedFS_interval_(
                        ns, nt, a, b, Xtildeinloop, Ytildeinloop, 
                        Sigmatilde_deltaz, SELECTIONinloop, GAMMAdeltaz
                    )

            elif method == 'BS':
                if k_type == 'stopping':
                    # Tương đương para_DA_BSwith_stoppingCriteria
                    if criterion == 'AIC':
                        SELECTIONinloop = BackwardSelection.SelectionAICforBS(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
                    elif criterion == 'BIC':
                        SELECTIONinloop = BackwardSelection.SelectionBIC(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
                    elif criterion == 'Adjusted R2':
                        SELECTIONinloop = BackwardSelection.SelectionAdjR2(Ytildeinloop, Xtildeinloop)
                    
                    intervalinloop = overconditioningBS.OC_DA_BS_Criterion_(
                        ns, nt, a, b, Xtildeinloop, Ytildeinloop, 
                        Sigmatilde_deltaz, SELECTIONinloop, GAMMAdeltaz, seed, criterion
                    )
                else:
                    # Tương đương para_DA_BS
                    SELECTIONinloop = BackwardSelection.fixedBS(Ytildeinloop, Xtildeinloop, len(SELECTION_F))[0]
                    intervalinloop, _, _ = overconditioningBS.OC_fixedBS_interval_(
                        ns, nt, a, b, Xtildeinloop, Ytildeinloop, 
                        Sigmatilde_deltaz, SELECTIONinloop, GAMMAdeltaz
                    )
            for left, right in intervalinloop:
                if left <= z <= right:
                    intervalinloop = [(left, right)]
                    break
            intervalinloop = intersection.interval_intersection(intervalinloop, itvDA)
            # print("Inloop itv:", intervalinloop)
            TD = intersection.interval_union(TD, intervalinloop)
            z = intervalinloop[-1][1] + 0.00001
    return TD