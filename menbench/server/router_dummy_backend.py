from menbench.server.base import RouterServer, ROUTER
from typing import List, Tuple
import math
import random

from menbench.data import Sample, Result

@ROUTER.register("OracleRouterServer")
class OracleRouterServer(RouterServer):
    def __init__(self, config):
        super().__init__(config)
    def run(self, result:dict[str, Result], ok)->Tuple[float, float] | None:
            best = math.inf
            for m in self.config["models"]:
                if ok[m] == 1:
                    best = min(best, result[m].record.latency)
            return (best, 1.0) if math.isfinite(best) else None

@ROUTER.register("MinRouterServer")
class MinRouterServer(RouterServer):
    def __init__(self, config):
        super().__init__(config)

    def run(self, result:dict[str, Result], ok)->Tuple[float, float] | None:
            return (result[self.config["min_model"]].record.latency, 1.0) if ok[self.config["min_model"]] else None

@ROUTER.register("MaxRouterServer")
class MaxRouterServer(RouterServer):
    def __init__(self, config):
        super().__init__(config)

    def run(self, result:dict[str, Result], ok)->Tuple[float, float] | None:
            return (result[self.config["max_model"]].record.latency, 1.0) if ok[self.config["max_model"]] else None

@ROUTER.register("RandomRouterServer")
class RandomRouterServer(RouterServer):
    def __init__(self, config):
        super().__init__(config)
    def run(self, result:dict[str, Result], ok)->Tuple[float, float] | None:
            # w = 1 / float(len(self.config["models"]))
            picked_model = random.choice(self.config["models"])
            return (result[picked_model].record.latency, 1.0) if ok[picked_model] else None