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
    detectedinter = []
    z = z_min
    zmax = z_max
    
    while z < zmax:
        z += 0.0001

        # Skip qua các khoảng đã được detect để tăng tốc độ xử lý
        for i in range(len(detectedinter)):
            if detectedinter[i][0] <= z <= detectedinter[i][1]:
                z = detectedinter[i][1] + 0.0001
                detectedinter = detectedinter[i:]
                break
        
        if z > zmax:
            break

        # 1. Cập nhật dữ liệu theo tham số z
        Ydeltaz = a + b * z
        XsXt_deltaz = np.concatenate((X, Ydeltaz), axis=1).copy()
        
        # 2. Giải Optimal Transport
        GAMMAdeltaz, basis_var_deltaz = OptimalTransport.solveOT(ns, nt, S_, h_, XsXt_deltaz).values()
        # GAMMAdeltaz = res_ot['gamma'] # Giả định cấu trúc trả về từ values()
        # basis_var_deltaz = res_ot['basis_var']

        Xtildeinloop = np.dot(GAMMAdeltaz, X)
        Ytildeinloop = np.dot(GAMMAdeltaz, Ydeltaz)
        Sigmatilde_deltaz = GAMMAdeltaz.T.dot(Sigma.dot(GAMMAdeltaz))

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
                
                intervalinloop = overconditioning.OC_Crit_interval(
                    ns, nt, a, b, XsXt_deltaz, Xtildeinloop, Ytildeinloop, 
                    Sigmatilde_deltaz, basis_var_deltaz, S_, h_, SELECTIONinloop, GAMMAdeltaz, criterion
                )
            else:
                # Tương đương para_DA_FSwithfixedK
                SELECTIONinloop = ForwardSelection.fixedSelection(Ytildeinloop, Xtildeinloop, len(SELECTION_F))[0]
                intervalinloop, _, _ = overconditioning.OC_fixedFS_interval(
                    ns, nt, a, b, XsXt_deltaz, Xtildeinloop, Ytildeinloop, 
                    Sigmatilde_deltaz, basis_var_deltaz, S_, h_, SELECTIONinloop, GAMMAdeltaz
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
                
                intervalinloop = overconditioningBS.OC_DA_BS_Criterion(
                    ns, nt, a, b, XsXt_deltaz, Xtildeinloop, Ytildeinloop, 
                    Sigmatilde_deltaz, basis_var_deltaz, S_, h_, SELECTIONinloop, GAMMAdeltaz, seed, criterion
                )
            else:
                # Tương đương para_DA_BS
                SELECTIONinloop = BackwardSelection.fixedBS(Ytildeinloop, Xtildeinloop, len(SELECTION_F))[0]
                intervalinloop, _, _ = overconditioningBS.OC_fixedBS_interval(
                    ns, nt, a, b, XsXt_deltaz, Xtildeinloop, Ytildeinloop, 
                    Sigmatilde_deltaz, basis_var_deltaz, S_, h_, SELECTIONinloop, GAMMAdeltaz
                )

        # 4. Hợp nhất các khoảng (Interval Union)
        detectedinter = intersection.interval_union(detectedinter, intervalinloop)

        # 5. Kiểm tra nếu Selection khớp với Selection gốc (F)
        if sorted(SELECTIONinloop) == sorted(SELECTION_F):
            TD = intersection.interval_union(TD, intervalinloop)

    return TD