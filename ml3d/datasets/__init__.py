from .semantickitti import SemanticKITTI
from .s3dis import S3DIS
from .parislille3d import ParisLille3D
from .toronto3d import Toronto3D
from .customdataset import Custom3D
from .semantic3d import Semantic3D
__all__ = [
    'SemanticKITTI', 'S3DIS', 'Toronto3D', 'ParisLille3D', 'Semantic3D',
    'Custom3D'
]
