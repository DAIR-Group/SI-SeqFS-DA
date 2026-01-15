from . import overconditioning
from . import overconditioningBS
import numpy as np
from . import OptimalTransport
from . import BackwardSelection 
from . import ForwardSelection
from . import intersection
from scipy.optimize import linprog

def find_valid_basis(H, x_nonzero_indices):
    """Tìm một tập chỉ số sao cho ma trận H[:, basis] là khả nghịch (Full Rank)"""
    num_constraints = H.shape[0]
    basis = list(x_nonzero_indices)
    
    # Sử dụng khử Gauss (hoặc QR) để tìm các cột độc lập tuyến tính bổ sung
    # Ở đây dùng phương pháp đơn giản: thử từng cột
    for i in range(H.shape[1]):
        if len(basis) == num_constraints:
            break
        if i not in basis:
            test_basis = basis + [i]
            if np.linalg.matrix_rank(H[:, test_basis]) == len(test_basis):
                basis.append(i)
    return sorted(basis)

def solve_quadratic_roots(a, b, c, current_z, z_max):
    """Tìm nghiệm thực nhỏ nhất của az^2 + bz + c = 0 trong khoảng (current_z, z_max]"""
    eps = 1e-7
    roots = []
    
    if abs(a) < 1e-10: # Tuyến tính: bz + c = 0
        if abs(b) > 1e-10:
            roots.append(-c / b)
    else: # Bậc hai
        delta = b**2 - 4*a*c
        if delta >= 0:
            sqrt_delta = np.sqrt(delta)
            roots.append((-b + sqrt_delta) / (2*a))
            roots.append((-b - sqrt_delta) / (2*a))
            
    valid_roots = [r for r in roots if r > current_z + eps and r <= z_max + eps]
    return min(valid_roots) if valid_roots else None

def parametric_simplex_quadratic(ns, nt, H, h, u, v, w, z_min, z_max):
    H = np.array(H, dtype=float)
    h = np.array(h, dtype=float)
    u, v, w = np.array(u), np.array(v), np.array(w)
    print("Shape:", H.shape, h.shape, u.shape, v.shape, w.shape)
    h = h.flatten()
    u, v, w = u.flatten(), v.flatten(), w.flatten()
    num_constraints, num_vars = H.shape
    current_z = z_min
    results = []

    # 1. Tìm lời giải ban đầu tại z_min
    
    c_start = u + v * z_min + w * (z_min**2)
    # Sử dụng method='highs' để có độ chính xác tốt hơn
    res = linprog(c_start, A_ub = - np.identity(ns * nt), b_ub = np.zeros((ns * nt, 1)), A_eq=H, b_eq=h, method='simplex',options={'maxiter': 1000000})



    if not res.success:
        return [] # Trả về rỗng nếu không có lời giải khả thi

    # Xác định các biến cơ sở ban đầu (đảm bảo ma trận vuông và độc lập tuyến tính)
    # nonzero_x = np.where(res.x > 1e-8)[0]
    basis_indices = res.basis #find_valid_basis(H, nonzero_x)

    while current_z <= z_max + 1e-9:
        B = H[:, basis_indices]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            # Nếu xảy ra lỗi ma trận suy biến, cố gắng tìm basis khác từ x hiện tại
            break

        # 2. Tính hệ số Reduced Cost: RC(z) = A*z^2 + B*z + C
        cb_u, cb_v, cb_w = u[basis_indices], v[basis_indices], w[basis_indices]
        
        # Công thức: rc = c - cB * B^-1 * H
        pi_u = cb_u @ B_inv
        pi_v = cb_v @ B_inv
        pi_w = cb_w @ B_inv

        rc_u = u - pi_u @ H
        rc_v = v - pi_v @ H
        rc_w = w - pi_w @ H

        # 3. Tìm z_next (điểm mà một RC biến ngoài cơ sở trở thành < 0)
        next_z = z_max
        entering_var = None
        
        for i in range(num_vars):
            if i in basis_indices:
                continue
            
            # Giải phương trình RC_i(z) = 0
            root = solve_quadratic_roots(rc_w[i], rc_v[i], rc_u[i], current_z, z_max)
            if root is not None and root < next_z:
                next_z = root
                entering_var = i

        # Lưu lại khoảng ổn định
        results.append((current_z, next_z, list(np.sort(basis_indices))))

        if entering_var is None or next_z >= z_max:
            break

        # 4. Thực hiện Pivot (Rời khỏi cơ sở)
        # Tính hướng di chuyển d = B^-1 * H_entering
        d = B_inv @ H[:, entering_var]
        x_B = B_inv @ h
        
        # Tỷ số Min-ratio để tìm biến rời khỏi cơ sở
        min_ratio = float('inf')
        leaving_idx_in_basis = -1
        for i in range(len(d)):
            if d[i] > 1e-9:
                ratio = x_B[i] / d[i]
                if ratio < min_ratio:
                    min_ratio = ratio
                    leaving_idx_in_basis = i
        
        if leaving_idx_in_basis == -1: # Không tìm được biến rời đi (unbounded)
            break
            
        basis_indices[leaving_idx_in_basis] = entering_var
        current_z = next_z

    return results

def compute_DA_intervals(ns, nt, a, b, c_, S, h, m_min=-20, m_max=20):
    OMEGA = OptimalTransport.constructOMEGA(ns, nt)

    Omega_a = OMEGA.dot(a)
    Omega_b = OMEGA.dot(b)

    w_tilde = c_ + Omega_a * Omega_a
    r_tilde = 2*Omega_a * Omega_b
    o_tilde = Omega_b * Omega_b

    intervals = parametric_simplex_quadratic(ns, nt,
        S, h,
        w_tilde, r_tilde, o_tilde,
        z_min=m_min,
        z_max=m_max
    )
    return intervals
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

    OMEGA = OptimalTransport.constructOMEGA(ns, nt)
    c_ = np.zeros((ns * nt, 1))
    for i in range(X.shape[1]):
        c_ += (OMEGA.dot(X[:, [i]])) * (OMEGA.dot(X[:, [i]]))
    list_da_itv = compute_DA_intervals(ns, nt, a, b, c_, S_, h_, m_min=z_min+0.0001, m_max=z_max)
    for ml, mr, B in list_da_itv:
        # print(f"Processing interval [{ml}, {mr}], Basis: {len(B)}, {[int(x) for x in B]}")
        z = ml + 0.00001
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
                    z = min(right, mr) + 0.00001
                    break
            TD = intersection.interval_union(TD, 
                                            intersection.interval_intersection(intervalinloop, itvDA))
            
    
    
    
    # while z < zmax:
    #     z += 0.0001
    #     print("z", z)
    #     # F(z) -> intervalinloop = [left, right] -------------
    #     # 1. Cập nhật dữ liệu theo tham số z
    #     Ydeltaz = a + b * z
    #     XsXt_deltaz = np.concatenate((X, Ydeltaz), axis=1).copy()
        
    #     # 2. Giải Optimal Transport
    #     GAMMAdeltaz, basis_var_deltaz = OptimalTransport.solveOT(ns, nt, S_, h_, XsXt_deltaz).values()
    #     # GAMMAdeltaz = res_ot['gamma'] # Giả định cấu trúc trả về từ values()
    #     # basis_var_deltaz = res_ot['basis_var']
    #     print(f"basis_var_deltaz: {len(basis_var_deltaz)}- {[int(x) for x in basis_var_deltaz]}")
    #     Xtildeinloop = np.dot(GAMMAdeltaz, X)
    #     Ytildeinloop = np.dot(GAMMAdeltaz, Ydeltaz)
    #     Sigmatilde_deltaz = GAMMAdeltaz.T.dot(Sigma.dot(GAMMAdeltaz))

    #     # 3. Lựa chọn Model (Selection) và Tính toán Interval
    #     if method == 'FS':
    #         if k_type == 'stopping':
    #             # Tương đương para_DA_FSwithStoppingCriterion
    #             if criterion == 'AIC':
    #                 SELECTIONinloop = ForwardSelection.SelectionAIC(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
    #             elif criterion == 'BIC':
    #                 SELECTIONinloop = ForwardSelection.SelectionBIC(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
    #             elif criterion == 'Adjusted R2':
    #                 SELECTIONinloop = ForwardSelection.SelectionAdjR2(Ytildeinloop, Xtildeinloop)
                
    #             intervalinloop = overconditioning.OC_Crit_interval(
    #                 ns, nt, a, b, XsXt_deltaz, Xtildeinloop, Ytildeinloop, 
    #                 Sigmatilde_deltaz, basis_var_deltaz, S_, h_, SELECTIONinloop, GAMMAdeltaz, criterion
    #             )
    #         else:
    #             # Tương đương para_DA_FSwithfixedK
    #             SELECTIONinloop = ForwardSelection.fixedSelection(Ytildeinloop, Xtildeinloop, len(SELECTION_F))[0]
    #             intervalinloop, _, _ = overconditioning.OC_fixedFS_interval(
    #                 ns, nt, a, b, XsXt_deltaz, Xtildeinloop, Ytildeinloop, 
    #                 Sigmatilde_deltaz, basis_var_deltaz, S_, h_, SELECTIONinloop, GAMMAdeltaz
    #             )

    #     elif method == 'BS':
    #         if k_type == 'stopping':
    #             # Tương đương para_DA_BSwith_stoppingCriteria
    #             if criterion == 'AIC':
    #                 SELECTIONinloop = BackwardSelection.SelectionAICforBS(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
    #             elif criterion == 'BIC':
    #                 SELECTIONinloop = BackwardSelection.SelectionBIC(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
    #             elif criterion == 'Adjusted R2':
    #                 SELECTIONinloop = BackwardSelection.SelectionAdjR2(Ytildeinloop, Xtildeinloop)
                
    #             intervalinloop = overconditioningBS.OC_DA_BS_Criterion(
    #                 ns, nt, a, b, XsXt_deltaz, Xtildeinloop, Ytildeinloop, 
    #                 Sigmatilde_deltaz, basis_var_deltaz, S_, h_, SELECTIONinloop, GAMMAdeltaz, seed, criterion
    #             )
    #         else:
    #             # Tương đương para_DA_BS
    #             SELECTIONinloop = BackwardSelection.fixedBS(Ytildeinloop, Xtildeinloop, len(SELECTION_F))[0]
    #             intervalinloop, _, _ = overconditioningBS.OC_fixedBS_interval(
    #                 ns, nt, a, b, XsXt_deltaz, Xtildeinloop, Ytildeinloop, 
    #                 Sigmatilde_deltaz, basis_var_deltaz, S_, h_, SELECTIONinloop, GAMMAdeltaz
    #             )
    #     for left, right in intervalinloop:
    #         if left <= z <= right:
    #             intervalinloop = [(left, right)]
    #             break
    #     # End F(z) -> intervalinloop = [left, right] -------------

    #     # 5. Kiểm tra nếu Selection khớp với Selection gốc (F) và Hợp nhất các khoảng hợp lệ (Interval Union)
    #     if sorted(SELECTIONinloop) == sorted(SELECTION_F):
    #         TD = intersection.interval_union(TD, intervalinloop)
    #     z = intervalinloop[-1][1]

    return TD