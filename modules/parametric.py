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
    
    if abs(a) < 1e-8: # Tuyến tính: bz + c = 0
        if abs(b) > 1e-8:
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
    # print("Starting parametric simplex with quadratic cost...")
    H = np.array(H, dtype=float)
    h = np.array(h, dtype=float)
    u, v, w = np.array(u), np.array(v), np.array(w)
    # print("Shape:", H.shape, h.shape, u.shape, v.shape, w.shape)
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
        results.append((current_z, next_z, np.sort(basis_indices)))

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

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

def para_DA_FSwithStoppingCriterion(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F, k, z_min=-20, z_max=20, seed=0):
    TD = []
    
    # --- PRE-COMPUTE DA BASIS INTERVALS ---
    OMEGA = OptimalTransport.constructOMEGA(ns, nt)
    c_ = np.zeros((ns * nt, 1))
    for i in range(X.shape[1]):
        c_ += (OMEGA.dot(X[:, [i]])) * (OMEGA.dot(X[:, [i]]))
    
    list_da_itv = compute_DA_intervals(ns, nt, a, b, c_, S_, h_, m_min=z_min, m_max=z_max)
    
    for ml, mr, B in list_da_itv:
        z = ml
        while z < mr:
            Ydeltaz = a + b * z
            
            # Reconstruct GAMMA from the constant basis B
            T = np.zeros(ns * nt)
            x_B = np.linalg.solve(S_[:, B], h_)
            T[B] = x_B.flatten()
            T = T.reshape((ns, nt))
            GAMMAdeltaz = OptimalTransport.constructGamma(ns, nt, T)
            
            # Map data to current domain
            Xtildeinloop = np.dot(GAMMAdeltaz, X)
            Ytildeinloop = np.dot(GAMMAdeltaz, Ydeltaz)
            Sigmatilde_deltaz = GAMMAdeltaz.T.dot(Sigma.dot(GAMMAdeltaz))
            
            # 1. Determine the Selection based on Criterion k
            if k == 'AIC':
                SELECTIONinloop = ForwardSelection.SelectionAIC(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
            elif k == 'BIC':
                SELECTIONinloop = ForwardSelection.SelectionBIC(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
            elif k == 'Adjusted R2':
                SELECTIONinloop = ForwardSelection.SelectionAdjR2(Ytildeinloop, Xtildeinloop)
            
            # 2. Calculate the components of the interval
            lst_SELECk, lst_P = ForwardSelection.list_residualvec(Xtildeinloop, Ytildeinloop)
            a_proj = GAMMAdeltaz.dot(a)
            b_proj = GAMMAdeltaz.dot(b)
            
            # Selection Interval (SFS)
            itvFS = overconditioning.interval_SFS(Xtildeinloop, Ytildeinloop, len(SELECTIONinloop), 
                                 lst_SELECk, lst_P, a_proj, b_proj)
            
            # Stopping Criterion Interval
            if k == 'AIC':
                itvCrit = overconditioning.interval_AIC(Xtildeinloop, Ytildeinloop, lst_P, 
                                                        len(SELECTIONinloop), a_proj, b_proj, 
                                                        Sigmatilde_deltaz, seed)
            elif k == 'BIC':
                itvCrit = overconditioning.interval_BIC(Xtildeinloop, Ytildeinloop, lst_P, 
                                                        len(SELECTIONinloop), a_proj, b_proj, 
                                                        Sigmatilde_deltaz, seed)
            elif k == 'Adjusted R2':
                itvCrit = overconditioning.interval_AdjustedR2(Xtildeinloop, Ytildeinloop, lst_P, 
                                                               len(SELECTIONinloop), a_proj, b_proj, 
                                                               Sigmatilde_deltaz, seed)

            # 3. Intersection: [itvFS] ∩ [itvCrit] ∩ [Current DA Range]
            intervalinloop = intersection.interval_intersection(itvFS, itvCrit)
            itvDA = [(ml, mr)]
            intervalinloop = intersection.interval_intersection(intervalinloop, itvDA)
            
            # 4. Check Match and Update TD
            if sorted(SELECTIONinloop) == sorted(SELECTION_F):
                TD = intersection.interval_union(TD, intervalinloop)
            
            # 5. Jump logic
            if not intervalinloop:
                z = mr  # Jump to next DA basis
            else:
                z = intervalinloop[-1][1] + 0.0001
                
    return TD

def para_DA_FSwithfixedK(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F, z_min=-20, z_max=20):
    TD = []
    
    # --- PRE-COMPUTE DA BASIS INTERVALS ---
    OMEGA = OptimalTransport.constructOMEGA(ns, nt)
    c_ = np.zeros((ns * nt, 1))
    for i in range(X.shape[1]):
        c_ += (OMEGA.dot(X[:, [i]])) * (OMEGA.dot(X[:, [i]]))
    
    # Returns list of (ml, mr, B) where B is the constant OT basis
    list_da_itv = compute_DA_intervals(ns, nt, a, b, c_, S_, h_, m_min=z_min, m_max=z_max)
    
    for ml, mr, B in list_da_itv:
        z = ml
        while z < mr:
            Ydeltaz = a + b * z
            
            # Reconstruct GAMMA from basis B
            T = np.zeros(ns * nt)
            x_B = np.linalg.solve(S_[:, B], h_)
            T[B] = x_B.flatten()
            T = T.reshape((ns, nt))
            GAMMAdeltaz = OptimalTransport.constructGamma(ns, nt, T)
            
            # Map data
            Xtildeinloop = np.dot(GAMMAdeltaz, X)
            Ytildeinloop = np.dot(GAMMAdeltaz, Ydeltaz)
            
            # Selection logic
            SELECTIONinloop = ForwardSelection.fixedSelection(Ytildeinloop, Xtildeinloop, len(SELECTION_F))[0]
            
            # --- FS INTERVAL CALCULATION ONLY ---
            lst_SELECk, lst_P = ForwardSelection.list_residualvec(Xtildeinloop, Ytildeinloop)
            
            # Compute itvFS using the projection of 'a' and 'b' onto the current GAMMA
            itvFS = overconditioning.interval_SFS(Xtildeinloop, Ytildeinloop, 
                                 len(SELECTION_F),
                                 lst_SELECk, lst_P,
                                 GAMMAdeltaz.dot(a), GAMMAdeltaz.dot(b))
            
            # Intersect itvFS with the current DA interval [ml, mr]
            itvDA = [(ml, mr)]
            intervalinloop = intersection.interval_intersection(itvFS, itvDA)
            
            # Update TD if selection matches target
            if sorted(SELECTIONinloop) == sorted(SELECTION_F):
                TD = intersection.interval_union(TD, intervalinloop)
            
            # Jump logic
            if not intervalinloop:
                z = mr # Move to next DA basis if no FS intersection found
            else:
                z = intervalinloop[-1][1] + 0.0001
                
    return TD

def para_DA_BSwith_stoppingCriteria(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F, seed=0, typeCrit='AIC', z_min=-20, z_max=20):
    TD = []
    
    # --- PRE-COMPUTE DA BASIS INTERVALS ---
    OMEGA = OptimalTransport.constructOMEGA(ns, nt)
    c_ = np.zeros((ns * nt, 1))
    for i in range(X.shape[1]):
        c_ += (OMEGA.dot(X[:, [i]])) * (OMEGA.dot(X[:, [i]]))
    
    list_da_itv = compute_DA_intervals(ns, nt, a, b, c_, S_, h_, m_min=z_min, m_max=z_max)
    
    for ml, mr, B in list_da_itv:
        z = ml
        while z < mr:
            Ydeltaz = a + b * z
            
            # Reconstruct GAMMA from the constant basis B
            T = np.zeros(ns * nt)
            x_B = np.linalg.solve(S_[:, B], h_)
            T[B] = x_B.flatten()
            T = T.reshape((ns, nt))
            GAMMAdeltaz = OptimalTransport.constructGamma(ns, nt, T)
            
            # Map data
            Xtildeinloop = np.dot(GAMMAdeltaz, X)
            Ytildeinloop = np.dot(GAMMAdeltaz, Ydeltaz)
            Sigmatilde_deltaz = GAMMAdeltaz.T.dot(Sigma.dot(GAMMAdeltaz))
            
            # 1. Determine Selection based on Criterion
            if typeCrit == 'AIC':
                SELECTIONinloop = BackwardSelection.SelectionAICforBS(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
            elif typeCrit == 'BIC':
                SELECTIONinloop = BackwardSelection.SelectionBIC(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
            elif typeCrit == 'Adjusted R2':
                SELECTIONinloop = BackwardSelection.SelectionAdjR2(Ytildeinloop, Xtildeinloop)
            
            # 2. Get residuals and projections
            lst_SELECk, lst_P = BackwardSelection.list_residualvec_BS(Xtildeinloop, Ytildeinloop)
            lst_SELECk.reverse() # Align with SBS logic
            
            a_proj = GAMMAdeltaz.dot(a)
            b_proj = GAMMAdeltaz.dot(b)
            k_len = len(SELECTIONinloop)
            
            # 3. Calculate BS Interval
            itvBS = overconditioningBS.interval_SBS(Xtildeinloop, Ytildeinloop, k_len, 
                                                    lst_SELECk, a_proj, b_proj)
            
            # 4. Calculate Criterion Interval
            if typeCrit == 'AIC':
                itvCrit = overconditioningBS.interval_AIC_BS(Xtildeinloop, Ytildeinloop, lst_P, k_len, 
                                                              a_proj, b_proj, Sigmatilde_deltaz, seed)
            elif typeCrit == 'BIC':
                itvCrit = overconditioningBS.interval_BIC(Xtildeinloop, Ytildeinloop, lst_P, k_len, 
                                                           a_proj, b_proj, Sigmatilde_deltaz, seed)
            elif typeCrit == 'Adjusted R2':
                itvCrit = overconditioningBS.interval_AdjustedR2(Xtildeinloop, Ytildeinloop, lst_P, k_len, 
                                                                  a_proj, b_proj, Sigmatilde_deltaz, seed)
            
            # 5. Intersection: [BS] ∩ [Crit] ∩ [Current DA Range]
            intervalinloop = intersection.interval_intersection(itvBS, itvCrit)
            itvDA = [(ml, mr)]
            intervalinloop = intersection.interval_intersection(intervalinloop, itvDA)
            
            # 6. Check Match and Update TD
            if sorted(SELECTIONinloop) == sorted(SELECTION_F):
                TD = intersection.interval_union(TD, intervalinloop)
            
            # 7. Jump logic
            if not intervalinloop:
                z = mr
            else:
                z = intervalinloop[-1][1] + 0.0001
                
    return TD

def para_DA_BS(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F, z_min=-20, z_max=20):
    TD = []
    
    # --- PRE-COMPUTE DA BASIS INTERVALS ---
    OMEGA = OptimalTransport.constructOMEGA(ns, nt)
    c_ = np.zeros((ns * nt, 1))
    for i in range(X.shape[1]):
        c_ += (OMEGA.dot(X[:, [i]])) * (OMEGA.dot(X[:, [i]]))
    
    # Get the list of (ml, mr, B) where B is the constant OT basis
    list_da_itv = compute_DA_intervals(ns, nt, a, b, c_, S_, h_, m_min=z_min, m_max=z_max)
    
    for ml, mr, B in list_da_itv:
        z = ml
        while z < mr:
            Ydeltaz = a + b * z
            
            # Reconstruct GAMMA from the constant basis B
            T = np.zeros(ns * nt)
            x_B = np.linalg.solve(S_[:, B], h_)
            T[B] = x_B.flatten()
            T = T.reshape((ns, nt))
            GAMMAdeltaz = OptimalTransport.constructGamma(ns, nt, T)
            
            # Map data
            Xtildeinloop = np.dot(GAMMAdeltaz, X)
            Ytildeinloop = np.dot(GAMMAdeltaz, Ydeltaz)
            
            # 1. Backward Selection (Fixed K)
            SELECTIONinloop = BackwardSelection.fixedBS(Ytildeinloop, Xtildeinloop, len(SELECTION_F))[0]
            
            # 2. Get residual vectors for BS
            # Note: Following your old code's logic to reverse the list
            lst_SELECk = BackwardSelection.list_residualvec_BS(Xtildeinloop, Ytildeinloop)[0]
            lst_SELECk.reverse()
            
            # 3. Calculate BS Interval only (no redundant DA)
            a_proj = GAMMAdeltaz.dot(a)
            b_proj = GAMMAdeltaz.dot(b)
            
            itvBS = overconditioningBS.interval_SBS(Xtildeinloop, Ytildeinloop, 
                                                    len(SELECTION_F),
                                                    lst_SELECk, a_proj, b_proj)
            
            # 4. Intersect itvBS with current DA range [ml, mr]
            itvDA = [(ml, mr)]
            intervalinloop = intersection.interval_intersection(itvBS, itvDA)
            
            # 5. Check match and update TD
            if sorted(SELECTIONinloop) == sorted(SELECTION_F):
                TD = intersection.interval_union(TD, intervalinloop)
            
            # 6. Jump logic
            if not intervalinloop:
                z = mr  # Jump to next DA basis if no intersection
            else:
                z = intervalinloop[-1][1] + 0.0001
                
    return TD
