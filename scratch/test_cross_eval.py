import numpy as np

def check_human_cross(kpts):
    """Robust human cross detection:
    Detects true human arm/wrist crossing (overhead or chest X),
    while rejecting artificial crosses (parallel raised hands, random proximity).
    """
    if kpts is None or len(kpts) < 11:
        return False, "insufficient_keypoints"
        
    nose = kpts[0]
    sh_l = kpts[5]
    sh_r = kpts[6]
    el_l = kpts[7] if len(kpts) > 7 else [0, 0, 0]
    el_r = kpts[8] if len(kpts) > 8 else [0, 0, 0]
    w_l = kpts[9]
    w_r = kpts[10]
    
    # Wrists must be detected with reasonable confidence
    if w_l[2] < 0.25 or w_r[2] < 0.25:
        return False, "low_wrist_conf"
        
    # Shoulders must be detected
    if sh_l[2] < 0.25 or sh_r[2] < 0.25:
        return False, "low_shoulder_conf"
        
    sh_width = max(15.0, abs(sh_r[0] - sh_l[0]))
    sh_y = (sh_l[1] + sh_r[1]) / 2.0
    sh_dx = sh_r[0] - sh_l[0]  # positive if sh_r is to the right in image
    wrist_dx = w_r[0] - w_l[0]
    
    # Head reference level
    if nose[2] > 0.25:
        head_y = nose[1]
    else:
        # Anatomically head center is ~ 0.5 shoulder widths above shoulders
        head_y = sh_y - (sh_width * 0.45)
        
    # 1. Height check:
    # A. Overhead cross: wrists are at or above head level
    overhead = (w_l[1] <= head_y + 15) and (w_r[1] <= head_y + 15) and (w_l[1] < sh_y - 10) and (w_r[1] < sh_y - 10)
    
    # B. Chest/torso cross: wrists are raised in front of chest/neck
    # Above elbow/hip level, around shoulder level
    in_front_chest = (w_l[1] < sh_y + sh_width * 0.8) and (w_r[1] < sh_y + sh_width * 0.8) and (w_l[1] > head_y - 20) and (w_r[1] > head_y - 20)
    
    if not (overhead or in_front_chest):
        return False, "not_raised_height"
        
    # 2. Geometric Cross Verification (detect humanly cross vs artificial parallel hands):
    # For a real human cross:
    # A. The wrists cross over each other relative to shoulders:
    #    If sh_r is right of sh_l (sh_dx > 0), then crossed means w_r is left of w_l (wrist_dx <= 0, or close to 0).
    #    Specifically, sh_dx * wrist_dx <= 0 (inversion), OR
    # B. Line segment intersection between arms:
    #    Segment from (L shoulder/elbow to L wrist) intersects (R shoulder/elbow to R wrist).
    
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
    def seg_intersect(A, B, C, D):
        return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))
        
    # Check arm intersection:
    # Primary: elbow-to-wrist if elbows available
    intersect = False
    if el_l[2] > 0.25 and el_r[2] > 0.25:
        intersect = seg_intersect(el_l[:2], w_l[:2], el_r[:2], w_r[:2])
    # Fallback / reinforcement: shoulder-to-wrist
    if not intersect:
        intersect = seg_intersect(sh_l[:2], w_l[:2], sh_r[:2], w_r[:2])
        
    # Horizontal wrist inversion check:
    # When crossed, wrists pass past each other horizontally
    is_inverted = (sh_dx * wrist_dx <= 0)
    
    # Distance between wrists:
    wrist_dist = np.hypot(w_l[0] - w_r[0], w_l[1] - w_r[1])
    wrists_touching_or_close = wrist_dist < (sh_width * 0.85)
    
    # A true human cross satisfies:
    # (1) Wrists are close/crossing (not far apart like a 'V' shape), AND
    # (2) Arm segments physically intersect OR wrists are horizontally inverted past each other!
    # Parallel hands (where wrist_dx has same sign as sh_dx and distance > 10% sh_width) are REJECTED as artificial!
    is_parallel_hands = (sh_dx * wrist_dx > 0) and not intersect and (abs(wrist_dx) > sh_width * 0.15)
    
    if is_parallel_hands:
        return False, "rejected_parallel_hands"
        
    if wrists_touching_or_close and (intersect or is_inverted):
        gesture_type = "WRISTS_CROSSED_OVERHEAD" if overhead else "ARMS_CROSSED_CHEST"
        return True, gesture_type
        
    return False, "no_cross"

# Test 1: test_gesture.py test case
k1 = np.zeros((17, 3), dtype=np.float32)
k1[0] = [320, 150, 0.9]
k1[5] = [300, 200, 0.9]
k1[6] = [340, 200, 0.9]
k1[9] = [325, 120, 0.9]
k1[10] = [315, 120, 0.9]
res1, t1 = check_human_cross(k1)
print("Test 1 (test_gesture.py):", res1, t1)

# Test 2: Parallel hands up (should be False)
k2 = np.copy(k1)
k2[9] = [310, 120, 0.9]
k2[10] = [330, 120, 0.9]
res2, t2 = check_human_cross(k2)
print("Test 2 (Parallel hands):", res2, t2)

# Test 3: Chest X cross
k3 = np.zeros((17, 3), dtype=np.float32)
k3[0] = [320, 150, 0.9]
k3[5] = [300, 200, 0.9]
k3[6] = [340, 200, 0.9]
k3[7] = [290, 240, 0.9]
k3[8] = [350, 240, 0.9]
k3[9] = [330, 190, 0.9] # L wrist crossed to right
k3[10] = [310, 190, 0.9] # R wrist crossed to left
res3, t3 = check_human_cross(k3)
print("Test 3 (Chest X cross):", res3, t3)
