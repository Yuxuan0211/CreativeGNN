from .fno.model import GraphFNO
from .geo_fno.model import GeoFNOProxy
from .gnn.model import StandardGNN
from .meshgraphnet.model import MeshGraphNet
from .mpnn.model import MPNN
from .pinn.model import PINN
from .regional_gnn.model import RegionalGNN

__all__ = ["GraphFNO", "GeoFNOProxy", "StandardGNN", "MeshGraphNet", "MPNN", "PINN", "RegionalGNN"]
