'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import dataclasses
from typing import (
    List,
)

@dataclasses.dataclass
class Runlist:
    input_datasets: List[str] = None
    missing_datasets: List[str] = None
    completed_datasets: List[str] = None

    stages: List[str] = None
    run_name: str = None
    replicate: int = None
