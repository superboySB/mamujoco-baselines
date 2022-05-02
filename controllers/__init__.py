REGISTRY = {}

from .basic_controller import BasicMAC
from .cqmix_controller import CQMixMAC
from .non_shared_controller import NonSharedMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["cqmix_mac"] = CQMixMAC