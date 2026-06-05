import math

from pydantic import BaseModel, field_validator
from typing import Any, Optional

from common.constant import EMOTIONS

class TeacherResponse(BaseModel):
    emotion: list[float]
    visual_cue: Optional[str]
    audio_cue: Optional[str]
    text_cue: Optional[str]
    language: Optional[str]
    number_of_faces: int

    @field_validator("emotion")
    @classmethod
    def emotion_validator(cls, v: Any) -> list[float]:
        if not isinstance(v, (list, tuple)):
            raise ValueError("Input must be a list or tuple")

        coerced_list: list[float] = []

        # Try converting each element to a float first, and push 0 if None
        for index, item in enumerate(v):
            try:
                converted_float = float(item) if item is not None else 0.0
            except (ValueError, TypeError):
                raise ValueError(f"Distribution for {EMOTIONS[index]} ('{item}') cannot be converted to a float")

            # Check if each element is between 0 and 1
            if not (0.0 <= converted_float <= 1.0):
                raise ValueError(f"Distribution for {EMOTIONS[index]} ('{item}') must be between 0 and 1")

            coerced_list.append(converted_float)

        # Check that the summation of all elements equals exactly 1
        if not math.isclose(math.fsum(coerced_list), 1.0, abs_tol=1e-9):
            raise ValueError(f"The sum of all elements must be exactly 1.0 (got {math.fsum(coerced_list)})")

        return coerced_list

    @field_validator("text_cue", "visual_cue", "audio_cue")
    @classmethod
    def cue_validator(cls, v: Any) -> str | None:
        if v is None:
            return None
        elif v == "None":
            return None
        return str(v)

    @field_validator("language")
    @classmethod
    def language_validator(cls, v: Any) -> str | None:
        if v is None:
            return None
        elif v in ["en"]:
            return "English"
        return str(v)

    @field_validator("number_of_faces")
    @classmethod
    def number_of_face_validator(cls, v: Any) -> int:
        if v is None:
            return 0
        return int(v)

