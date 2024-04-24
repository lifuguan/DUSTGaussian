from .realestate import *
from .realestate import *
from .llff import *
from .llff_test import *
from .ibrnet_collected import *
from .realestate import *
from .llff_render import *

dataset_dict = {
    "realestate": RealEstateDataset,
    "llff": LLFFDataset,
    "ibrnet_collected": IBRNetCollectedDataset,
    "llff_test": LLFFTestDataset,
    "llff_render": LLFFRenderDataset,
}
