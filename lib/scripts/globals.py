from enum import Enum

classifier              = ""
model                   = ""
data_visuals_dir        = ""
filter_visuals_dir      = ""
layer_visuals_dir       = ""
log_dir                 = ""
training_masks_dir      = ""
initialized             = True

# an enumerated class and map for keeping tracking of available types of datasets that we are allowed to create
class DB_TYPES(Enum):
    TRAIN       = 0
    TEST        = 1
    VALIDATION  = 2

DB_MAP = {}
DB_MAP[DB_TYPES.TRAIN]          = 'train'
DB_MAP[DB_TYPES.TEST]           = 'test'
DB_MAP[DB_TYPES.VALIDATION]     = 'validation'

# an enumerated class and map for keeping tracking of dataset dimensions
# (patch, label, image_name, j, i, orientation_code, roof_indicator, roof_percentage)
class DB_DIM_TYPES(Enum):
    PATCH               = 0
    LABEL               = 1
    IMAGE_NAME          = 2
    J                   = 3
    I                   = 4
    ORIENTATION_CODE    = 5
    ROOF_INDICATOR      = 6
    ROOF_PERCENTAGE     = 7

DB_DIM_MAP = {}
DB_DIM_MAP[DB_DIM_TYPES.PATCH]              = 'patch'
DB_DIM_MAP[DB_DIM_TYPES.LABEL]              = 'label'
DB_DIM_MAP[DB_DIM_TYPES.IMAGE_NAME]         = 'image_name'
DB_DIM_MAP[DB_DIM_TYPES.J]                  = 'j'
DB_DIM_MAP[DB_DIM_TYPES.I]                  = 'i'
DB_DIM_MAP[DB_DIM_TYPES.ORIENTATION_CODE]   = 'orientation_code'
DB_DIM_MAP[DB_DIM_TYPES.ROOF_INDICATOR]     = 'roof_indicator'
DB_DIM_MAP[DB_DIM_TYPES.ROOF_PERCENTAGE]    = 'roof_percentage'



