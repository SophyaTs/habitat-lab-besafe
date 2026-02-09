import json
from habitat.core.utils import Singleton
import attr
from typing import (
    TYPE_CHECKING,
    Any,
    List,
)
import time

DANGEROUS1_THRESHOLD = 0.5
DANGEROUS2_THRESHOLD = 0.25
DANGEROUS3_THRESHOLD = 0.125
FLUSH_CHECKPOINT = 10

class EpisodeStats:
    def __init__(self, id: int, seq_id: int):
        self.id: int = id # id in dataset
        self.seq_id: int = seq_id # id in runtime
        self.dangerous1_steps: int = 0
        self.dangerous2_steps: int = 0
        self.dangerous3_steps: int = 0
        self.move_steps = 0
        self.steps: int = 0
        self.hits: List[List[float]] = []

    def submit_step(self, dco, move = False):
        if move:
            if dco < DANGEROUS3_THRESHOLD:
                self.dangerous3_steps += 1
            elif dco < DANGEROUS2_THRESHOLD:
                self.dangerous2_steps += 1
            elif dco < DANGEROUS1_THRESHOLD:
                self.dangerous1_steps += 1
            self.move_steps += 1

        self.steps += 1

    def submit_hit(self, pos):
        self.hits.append(pos.tolist())

    def submit_success(self, success):
        self.success = success


class ObjNavEpisodeStats(EpisodeStats):
    def __init__(self, id: int, seq_id: int, category: str):
        super().__init__(id, seq_id)
        self.category = category

@attr.s(auto_attribs=True, slots=True)
class StatisticsSingleton(metaclass=Singleton):
    data: List[EpisodeStats] = attr.ib(init=True, factory=list)
    offset: int = 0

    def set_offset(self, offset):
        self.offset += offset

    def get_seq_id(self):
        return self.offset + len(self.data)

    def add_episode(self, ep: EpisodeStats):
        self.data.append(ep)

    def submit_step(self, dco, move = False):
        self.data[-1].submit_step(dco, move)

    def submit_hit(self, pos):
        self.data[-1].submit_hit(pos)

    def submit_success(self, success):
        self.data[-1].submit_success(success)

    def finalize_episode(self, force_flush = False):
        if (len(self.data) > 0 and len(self.data) % FLUSH_CHECKPOINT == 0) or force_flush:
            print("Flushing episodes...")
            def serialize(obj):
                return obj.__dict__
            
            # jstr = json.dump(self.data, default=serialize)
            timestamp = int(time.time())
            with open(f"data/stats/{timestamp}_ep{self.data[0].seq_id}-ep{self.data[-1].seq_id}.json" , "w") as f:
                json.dump(self.data, f, default=serialize, indent=2)

Statistics: StatisticsSingleton = StatisticsSingleton()