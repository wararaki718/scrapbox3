from dataclasses import dataclass


@dataclass
class Morph:
    surface: str
    pos: str # pos=part of speech
    pos1: str
    pos2: str
    pos3: str
    conjugated: str
    conjugation: str
    base: str
    read: str
    pronunciation: str
