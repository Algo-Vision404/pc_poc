"""Class vocabulary shared by the dataset, the model, and the consumers.

Kept in its own module because ``data`` and ``model`` import each other's
constants (data needs ``DET_STRIDE``, model needs the class count) and a third
neutral module breaks that cycle.
"""

from __future__ import annotations

#: Detection classes. Each has a distinct silhouette and hue family.
CLASS_NAMES: tuple[str, ...] = ("block", "sphere", "triangle")
NUM_CLASSES: int = len(CLASS_NAMES)

#: Segmentation adds a background class at index 0.
NUM_SEG_CLASSES: int = NUM_CLASSES + 1

#: Distinct hue per class, used for the segmentation overlay.
SEG_PALETTE: tuple[tuple[int, int, int], ...] = (
    (148, 163, 184),  # 0 background — slate
    (239, 68, 68),    # 1 block — red
    (34, 197, 94),    # 2 sphere — green
    (59, 130, 246),   # 3 triangle — blue
)


def class_name(index: int) -> str:
    """Name of detection class ``index``.

    Deliberately permissive: a checkpoint trained with a different class count
    still decodes instead of raising at request time.
    """
    if 0 <= index < NUM_CLASSES:
        return CLASS_NAMES[index]
    return f"class_{index}"
