"""
Domain model for Painting entity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from uuid import UUID


@dataclass
class StyleScore:
    """Represents a style classification score."""
    estilo: str
    confianza: float


@dataclass
class Painting:
    """Domain model representing a painting in the catalog."""

    id: UUID
    archivo: str
    estilo_principal: str
    confianza: float
    ruta: Optional[str] = None
    estilo_2: Optional[str] = None
    confianza_2: Optional[float] = None
    estilo_3: Optional[str] = None
    confianza_3: Optional[float] = None
    todos_estilos: List[StyleScore] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    imagen: Optional[bytes] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def top_estilos(self) -> List[StyleScore]:
        """Returns top 3 styles."""
        styles = [StyleScore(self.estilo_principal, self.confianza)]
        if self.estilo_2 and self.confianza_2:
            styles.append(StyleScore(self.estilo_2, self.confianza_2))
        if self.estilo_3 and self.confianza_3:
            styles.append(StyleScore(self.estilo_3, self.confianza_3))
        return styles

    @classmethod
    def from_dict(cls, data: dict) -> "Painting":
        """Creates a Painting from a dictionary (database row)."""
        todos_estilos = []
        if data.get("todos_estilos"):
            raw_estilos = data["todos_estilos"]
            if isinstance(raw_estilos, list):
                todos_estilos = [
                    StyleScore(e["estilo"], e["confianza"])
                    for e in raw_estilos
                ]

        return cls(
            id=data["id"],
            archivo=data["archivo"],
            ruta=data.get("ruta"),
            estilo_principal=data["estilo_principal"],
            confianza=data["confianza"],
            estilo_2=data.get("estilo_2"),
            confianza_2=data.get("confianza_2"),
            estilo_3=data.get("estilo_3"),
            confianza_3=data.get("confianza_3"),
            todos_estilos=todos_estilos,
            embedding=data.get("embedding"),
            imagen=data.get("imagen"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )
