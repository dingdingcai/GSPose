cosim_topk = -1
refer_view_num = 8
DINO_PATCH_SIZE = 14
zoom_image_margin = 0
zoom_image_scale = 224
query_longside_scale = 672
query_shortside_scale = query_longside_scale * 3 // 4

coarse_threshold = 0.05
coarse_bbox_padding = 1.25

finer_threshold = 0.5
finer_bbox_padding = 1.25

enable_fine_detection = True

save_reference_mask = True
#### configure for 3D-GS-Refiner  ####
ROT_TOPK = 1   # single rotation proposal
MIN_ROT_DEGREE_DIST = 30
NUM_ROT_SAMPLING_BINS = 3

SO3_DIM = 4 # rotation quaternion
# SO3_DIM = 6 # rotation vectors
# SO3_DIM = 9 # rotation matrix

WARMUP = 10
# LOSS_MS = 0.5  # weight for the SSIM loss
END_LR = 1e-6
START_LR = 5e-3
MAX_STEPS = 100
GS_RENDER_SIZE = 224
EARLY_STOP_MIN_STEPS = 5
EARLY_STOP_LOSS_GRAD_NORM = 1e-6

USE_MSE = True
USE_SSIM = True
USE_MS_SSIM = True

BINARIZE_MASK = False
USE_YOLO_BBOX = True
USE_ALLOCENTRIC = True
APPLY_ZOOM_AND_CROP = True
CC_INCLUDE_SUPMASK = False
