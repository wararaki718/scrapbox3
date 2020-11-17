from dataclasses import dataclass


@dataclass
class Morph:
    surface: str
    part_of_speech: str
    part_of_speech1: str
    part_of_speech2: str
    part_of_speech3: str
    inflected_type: str
    inflected_form: str
    base_form: str
    reading: str
    phonetic: str
