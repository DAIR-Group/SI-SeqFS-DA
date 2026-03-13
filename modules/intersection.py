import numpy as np

# def solve_quadratic_inequality(a, b, c,seed = 0):
#     """ ax^2 + bx +c <= 0 """
#     if abs(a) < 1e-7:
#         a = 0
#     if abs(b) < 1e-7:
#         b = 0
#     if abs(c) < 1e-7:
#         c = 0
#     if a == 0:
#         # print(f"b: {b}")
#         if b > 0:
#             # return [(-np.inf, -c / b)]
#             right = -c / b
#             return [(-np.inf, right)]
#             # return [(-np.inf, np.around(-c / b, 12))]
#         elif b == 0:
#             # print(f"c: {c}")
#             if c <= 0:
#                 return [(-np.inf, np.inf)]
#             else:
#                 print('Error bx + c', seed)
#                 return 
#         else:
#             # left = np.around(-c / b, 12)
#             left = -c / b
#             return [(left, np.inf)]
#     delta = b*b - 4*a*c
#     if delta < 0:
#         if a < 0:
#             return [(-np.inf, np.inf)]
#         else:
#             print("Error to find interval. ")
#     # print("delta:", delta)
#     # print(f"2a: {2*a}")
#     x1 = (- b - np.sqrt(delta)) / (2.0*a)
#     x2 = (- b + np.sqrt(delta)) / (2.0*a)
#     # if x1 > x2:
#     #     x1, x2 = x2, x1  
#     # x1 = np.around(x1, 12)
#     # x2 = np.around(x2, 12)
#     if a < 0:
#         return [(-np.inf, x2),(x1, np.inf)]
#     return [(x1,x2)]
def solve_quadratic_inequality(a, b, c):
    # Solve quadratic inequality has the form of ax^2 + bx + c <= 0
    if -1e-7 <= a <= 1e-7:
        a = 0.0

    if -1e-7 <= b <= 1e-7:
        b = 0.0

    if -1e-7 <= c <= 1e-7:
        c = 0.0

    if a == 0.0:
        if b != 0.0:
            root = np.round(-c/b, 12)

            if b > 0.0:
                return [(-np.inf, root)]
            else:
                return [(root, np.inf)]
        
        else:
            if c > 0.0:
                # print("Error no roots: c > 0")
                return None
            else:
                return [(-np.inf, np.inf)]
            
    delta = b**2 - 4 * a * c

    if delta < 0.0:
        if a > 0.0:
            # print("Error no roots: a > 0")
            return None
        else: 
            return [(-np.inf, np.inf)]
    else:
        sqrt_delta = np.sqrt(delta)

        if b > 0:
            root1 = np.round((-b-sqrt_delta)/(2*a), 12)
        else:
            root1 = np.round((-b+sqrt_delta)/(2*a), 12)

        root2 = np.round(c / (a * root1), 12)

        roots = tuple(np.sort([root1, root2]))
        
        if a > 0:
            return [roots]
        else:
            return [(-np.inf, roots[0]), (roots[1], np.inf)]
def interval_intersection(a, b):
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        a_start, a_end = a[i]
        b_start, b_end = b[j]
        
        # Calculate the potential intersection
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        
        # If the interval is valid, add to results
        if start < end:
            result.append((start, end))
        
        # Move the pointer which ends first
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return result
def interval_union(a, b):
    # Merge the two sorted interval lists into one sorted list
    merged = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] < b[j][0]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    # Add any remaining intervals from a or b
    merged.extend(a[i:])
    merged.extend(b[j:])
    
    # Merge overlapping intervals
    if not merged:
        return []
    
    result = [merged[0]]
    for current in merged[1:]:
        last = result[-1]
        if current[0] < last[1]:
            # Overlapping or adjacent, merge them
            new_start = last[0]
            new_end = max(last[1], current[1])
            result[-1] = (new_start, new_end)
        else:
            result.append(current)
    return result
