#!/usr/bin/env python3
"""
Load initial catalog data to PostgreSQL.

Usage:
    python -m scripts.load_initial_data [catalog.json]
    python -m scripts.load_initial_data test  # Run test queries
"""

import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from psycopg2.extras import execute_values

from src.database.connection import get_db_connection


def load_catalog(json_path: str = "catalogo_estilos.json"):
    """Load catalog JSON to the database."""

    with open(json_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    print(f"Loading {len(catalog)} paintings from {json_path}")

    conn = get_db_connection()
    if not conn:
        return

    cursor = conn.cursor()

    records = []
    for item in catalog:
        if "error" in item:
            print(f"  Skipping {item.get('archivo', 'unknown')} (has error)")
            continue

        top_estilos = item.get("top_estilos", [])

        record = (
            item["archivo"],
            item.get("ruta", ""),
            item["estilo_principal"],
            item["confianza"],
            top_estilos[1]["estilo"] if len(top_estilos) > 1 else None,
            top_estilos[1]["confianza"] if len(top_estilos) > 1 else None,
            top_estilos[2]["estilo"] if len(top_estilos) > 2 else None,
            top_estilos[2]["confianza"] if len(top_estilos) > 2 else None,
            json.dumps(item.get("todos_los_estilos", []))
        )
        records.append(record)

    insert_query = """
        INSERT INTO pinturas (
            archivo, ruta, estilo_principal, confianza,
            estilo_2, confianza_2, estilo_3, confianza_3,
            todos_estilos
        ) VALUES %s
        ON CONFLICT (archivo) DO UPDATE SET
            estilo_principal = EXCLUDED.estilo_principal,
            confianza = EXCLUDED.confianza,
            estilo_2 = EXCLUDED.estilo_2,
            confianza_2 = EXCLUDED.confianza_2,
            estilo_3 = EXCLUDED.estilo_3,
            confianza_3 = EXCLUDED.confianza_3,
            todos_estilos = EXCLUDED.todos_estilos,
            updated_at = CURRENT_TIMESTAMP
    """

    try:
        cursor.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'pinturas_archivo_key'
                ) THEN
                    ALTER TABLE pinturas ADD CONSTRAINT pinturas_archivo_key UNIQUE (archivo);
                END IF;
            END $$;
        """)

        execute_values(cursor, insert_query, records)
        conn.commit()
        print(f"Inserted {len(records)} paintings")

    except Exception as e:
        conn.rollback()
        print(f"Error inserting: {e}")

    cursor.execute("SELECT * FROM resumen_estilos")
    print("\nSUMMARY IN DATABASE:")
    print("-" * 40)
    for row in cursor.fetchall():
        estilo, cantidad, confianza = row
        print(f"  {estilo:18} {cantidad:3} paintings  (conf: {confianza})")

    cursor.close()
    conn.close()
    print("\nLoad completed")


def test_queries():
    """Run test queries."""
    conn = get_db_connection()
    if not conn:
        return

    cursor = conn.cursor()

    print("\nTEST QUERIES:")
    print("=" * 50)

    print("\n1. 'Urbano' style paintings:")
    cursor.execute(
        "SELECT archivo, confianza FROM pinturas WHERE estilo_principal = 'Urbano' LIMIT 5"
    )
    for row in cursor.fetchall():
        print(f"   - {row[0]} ({row[1]*100:.1f}%)")

    print("\n2. Paintings with confidence > 12%:")
    cursor.execute(
        "SELECT archivo, estilo_principal, confianza FROM pinturas WHERE confianza > 0.12 ORDER BY confianza DESC LIMIT 5"
    )
    for row in cursor.fetchall():
        print(f"   - {row[0]}: {row[1]} ({row[2]*100:.1f}%)")

    print("\n3. Total by style:")
    cursor.execute("SELECT * FROM resumen_estilos")
    for row in cursor.fetchall():
        print(f"   - {row[0]}: {row[1]}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_queries()
    else:
        json_file = sys.argv[1] if len(sys.argv) > 1 else "catalogo_estilos.json"
        load_catalog(json_file)
        print("\n" + "=" * 50)
        print("To run test queries: python -m scripts.load_initial_data test")
