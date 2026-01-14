import os
# Ép các thư viện toán học chỉ dùng 1 luồng mỗi tiến trình
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np # Import numpy sau khi đã set environ
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# Import các module bổ trợ
from . import overconditioning
from . import overconditioningBS
from . import OptimalTransport
from . import BackwardSelection 
from . import ForwardSelection
from . import intersection

# --- BIẾN TOÀN CỤC CHO WORKER ---
_worker_status = None

def _init_worker(shared_dict):
    global _worker_status
    _worker_status = shared_dict

def compute_interval_at_z(z, ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F, method, k_type, criterion, seed):
    """Logic F(z) - Giữ nguyên cơ chế gốc"""
    Ydeltaz = a + b * z
    XsXt_deltaz = np.concatenate((X, Ydeltaz), axis=1).copy()
    
    # Giải Optimal Transport
    GAMMAdeltaz, basis_var_deltaz = OptimalTransport.solveOT(ns, nt, S_, h_, XsXt_deltaz).values()

    Xtildeinloop = np.dot(GAMMAdeltaz, X)
    Ytildeinloop = np.dot(GAMMAdeltaz, Ydeltaz)
    Sigmatilde_deltaz = GAMMAdeltaz.T.dot(Sigma.dot(GAMMAdeltaz))

    if method == 'FS':
        if k_type == 'stopping':
            if criterion == 'AIC': SELECTIONinloop = ForwardSelection.SelectionAIC(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
            elif criterion == 'BIC': SELECTIONinloop = ForwardSelection.SelectionBIC(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
            else: SELECTIONinloop = ForwardSelection.SelectionAdjR2(Ytildeinloop, Xtildeinloop)
            
            intervalinloop = overconditioning.OC_Crit_interval(ns, nt, a, b, XsXt_deltaz, Xtildeinloop, Ytildeinloop, Sigmatilde_deltaz, basis_var_deltaz, S_, h_, SELECTIONinloop, GAMMAdeltaz, criterion)
        else:
            SELECTIONinloop = ForwardSelection.fixedSelection(Ytildeinloop, Xtildeinloop, len(SELECTION_F))[0]
            intervalinloop, _, _ = overconditioning.OC_fixedFS_interval(ns, nt, a, b, XsXt_deltaz, Xtildeinloop, Ytildeinloop, Sigmatilde_deltaz, basis_var_deltaz, S_, h_, SELECTIONinloop, GAMMAdeltaz)
    else: # BS
        if k_type == 'stopping':
            if criterion == 'AIC': SELECTIONinloop = BackwardSelection.SelectionAICforBS(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
            elif criterion == 'BIC': SELECTIONinloop = BackwardSelection.SelectionBIC(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
            else: SELECTIONinloop = BackwardSelection.SelectionAdjR2(Ytildeinloop, Xtildeinloop)
            
            intervalinloop = overconditioningBS.OC_DA_BS_Criterion(ns, nt, a, b, XsXt_deltaz, Xtildeinloop, Ytildeinloop, Sigmatilde_deltaz, basis_var_deltaz, S_, h_, SELECTIONinloop, GAMMAdeltaz, seed, criterion)
        else:
            SELECTIONinloop = BackwardSelection.fixedBS(Ytildeinloop, Xtildeinloop, len(SELECTION_F))[0]
            intervalinloop, _, _ = overconditioningBS.OC_fixedBS_interval(ns, nt, a, b, XsXt_deltaz, Xtildeinloop, Ytildeinloop, Sigmatilde_deltaz, basis_var_deltaz, S_, h_, SELECTIONinloop, GAMMAdeltaz)

    final_interval = None
    for left, right in intervalinloop:
        if left <= z <= right:
            final_interval = (left, right)
            break
            
    return SELECTIONinloop, final_interval

def _init_worker(shared_dict):
    global _worker_status
    _worker_status = shared_dict

def worker_task(real_worker_id, z_start, z_end, params):
    """Xử lý một đoạn z nhỏ và báo cáo tiến độ"""
    local_td = []
    z = z_start
    delta_z = 0.0001
    target_selection = np.sort(params[8])
    
    while z < z_end:
        current_z = z + delta_z
        if current_z >= z_end: break
        
        try:
            # # Logic tính toán chính (nhân ma trận bên trong giờ chỉ dùng 1 core)
            # from .utils import compute_interval_at_z # Giả sử hàm này nằm trong utils
            sel, interval = compute_interval_at_z(current_z, *params)
        except Exception:
            z = current_z
            _worker_status[real_worker_id] = (z - z_start, z, z_start, z_end)
            continue
        
        if interval is not None and np.array_equal(np.sort(sel), target_selection):
            actual_left = max(interval[0], current_z)
            actual_right = min(interval[1], z_end)
            local_td = intersection.interval_union(local_td, [(actual_left, actual_right)])
            z = max(actual_right, current_z + delta_z)
        else:
            z = current_z
        
        # Cập nhật trạng thái cho CPU tương ứng
        _worker_status[real_worker_id] = (z - z_start, z, z_start, z_end)
            
    return local_td

def parametric(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F, 
               method='FS', k_type='fixed', criterion='AIC', 
               z_min=-20, z_max=20, seed=0, num_workers=None):
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count() # Sẽ là 16 trên máy bạn

    # Chia cực nhỏ dải Z (400 task) để các nhân rảnh tay "nhảy vào" làm ngay
    num_sub_tasks = 400 
    edges = np.linspace(z_min, z_max, num_sub_tasks + 1)
    params = (ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F, method, k_type, criterion, seed)
    
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    
    for i in range(num_workers):
        shared_dict[i] = (0.0, 0.0, 0.0, 0.0)

    combined_td = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("• {task.completed:.3f}/{task.total:.3f} units"),
        TimeRemainingColumn(),
    ) as progress:
        
        # Tạo 16 thanh tiến trình cho 16 nhân thực
        worker_tids = [progress.add_task(f"CPU {i:2d}: Idle", total=1.0) for i in range(num_workers)]
        # Thanh tổng quát theo dõi 400 tasks
        overall_tid = progress.add_task("[bold magenta]TOTAL PROGRESS", total=num_sub_tasks)

        with ProcessPoolExecutor(max_workers=num_workers, 
                                 initializer=_init_worker, 
                                 initargs=(shared_dict,)) as executor:
            
            # Gửi 400 task nhỏ vào hàng đợi
            futures = [executor.submit(worker_task, i % num_workers, edges[i], edges[i+1], params) 
                       for i in range(num_sub_tasks)]
            
            while not all(f.done() for f in futures):
                for i in range(num_workers):
                    done_len, curr_z, s_edge, e_edge = shared_dict[i]
                    chunk_size = e_edge - s_edge
                    if chunk_size > 0:
                        progress.update(
                            worker_tids[i], 
                            completed=done_len, 
                            total=chunk_size, 
                            description=f"CPU {i:2d}: z={curr_z:>7.3f}"
                        )
                
                completed_count = sum(1 for f in futures if f.done())
                progress.update(overall_tid, completed=completed_count)
                time.sleep(0.3)

            for fut in futures:
                res = fut.result()
                if res:
                    combined_td = intersection.interval_union(combined_td, res)

    return combined_td