"""Таксономия меток (ТЗ §7.2): иерархия — отдельный версионируемый YAML.

В манифесте хранится САМАЯ СПЕЦИФИЧНАЯ доступная метка; предки выводятся
кодом отсюда и никогда не денормализуются в манифест.

Формат ``taxonomy.yaml``::

    levels: [coarse, fine]          # грубый -> тонкий
    nodes:
      malignant: {level: coarse}
      benign:    {level: coarse}
      melanoma:  {level: fine, parent: malignant}
      nevus:     {level: fine, parent: benign}

id внутри уровня — по алфавиту имён (детерминированы и воспроизводимы);
словарь уровня доступен через :attr:`Taxonomy.vocab`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from vision_lab.core.batch import MISSING_LABEL


@dataclass(frozen=True)
class Taxonomy:
    levels: tuple[str, ...]                      # имена уровней, грубый -> тонкий
    level_of: dict[str, str]                     # узел -> уровень
    parent: dict[str, str | None]                # узел -> родитель (None у корневых)
    vocab: dict[str, dict[str, int]]             # уровень -> {узел: id}

    # -- конструирование -----------------------------------------------------
    @classmethod
    def from_dict(cls, spec: dict) -> Taxonomy:
        levels = tuple(spec["levels"])
        nodes: dict[str, dict] = spec["nodes"]
        level_of, parent = {}, {}
        for name, meta in nodes.items():
            meta = meta or {}
            level = meta.get("level")
            if level not in levels:
                raise ValueError(f"Узел {name!r}: уровень {level!r} не объявлен в levels={levels}")
            level_of[name] = level
            parent[name] = meta.get("parent")

        for name, par in parent.items():
            if par is None:
                continue
            if par not in level_of:
                raise ValueError(f"Узел {name!r}: родитель {par!r} не описан в nodes")
            if levels.index(level_of[par]) >= levels.index(level_of[name]):
                raise ValueError(
                    f"Узел {name!r} (уровень {level_of[name]!r}): родитель {par!r} "
                    f"должен быть на более грубом уровне, а он на {level_of[par]!r}"
                )

        vocab = {
            level: {n: i for i, n in enumerate(sorted(n for n, lv in level_of.items() if lv == level))}
            for level in levels
        }
        return cls(levels=levels, level_of=level_of, parent=parent, vocab=vocab)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Taxonomy:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    # -- запросы ---------------------------------------------------------------
    def num_classes(self, level: str) -> int:
        return len(self.vocab[level])

    def ancestors(self, label: str) -> dict[str, str]:
        """{уровень: узел} для метки и всех её предков (сама метка включена)."""
        if label not in self.level_of:
            raise KeyError(f"Метка {label!r} отсутствует в таксономии")
        out: dict[str, str] = {}
        node: str | None = label
        while node is not None:
            out[self.level_of[node]] = node
            node = self.parent[node]
        return out

    def levels_vector(self, label: str | None) -> np.ndarray:
        """(L,) int64: id по каждому уровню; -1 — уровень тоньше метки или метки нет.

        Неизвестная (не-None) метка — ошибка, не тихие -1 (ТЗ §7.4: тихие
        фолбэки запрещены).
        """
        vec = np.full(len(self.levels), MISSING_LABEL, dtype=np.int64)
        if label is None:
            return vec
        for level, node in self.ancestors(label).items():
            vec[self.levels.index(level)] = self.vocab[level][node]
        return vec
