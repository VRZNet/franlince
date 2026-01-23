"""
=============================================================================
CARGAR CATÁLOGO A POSTGRESQL - FRANLINCE
=============================================================================

Requisitos:
    pip install psycopg2-binary

Uso:
    python load_to_db.py

=============================================================================
"""

import json
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path


# Configuración de la base de datos
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "franlince_catalog",
    "user": "franlince",
    "password": "franlince123"
}


def connect_db():
    """Conecta a la base de datos"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Conectado a PostgreSQL")
        return conn
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None


def load_catalog(json_path: str = "catalogo_estilos.json"):
    """Carga el catálogo JSON a la base de datos"""
    
    # Leer JSON
    with open(json_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    
    print(f"📂 Cargando {len(catalog)} pinturas desde {json_path}")
    
    # Conectar
    conn = connect_db()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    # Preparar datos
    records = []
    for item in catalog:
        if "error" in item:
            print(f"  ⚠️  Saltando {item.get('archivo', 'unknown')} (tiene error)")
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
    
    # Insertar
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
        # Agregar constraint único si no existe
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
        print(f"✅ Insertadas {len(records)} pinturas")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error al insertar: {e}")
    
    # Mostrar resumen
    cursor.execute("SELECT * FROM resumen_estilos")
    print("\n📊 RESUMEN EN BASE DE DATOS:")
    print("-" * 40)
    for row in cursor.fetchall():
        estilo, cantidad, confianza = row
        print(f"  {estilo:18} {cantidad:3} pinturas  (conf: {confianza})")
    
    cursor.close()
    conn.close()
    print("\n✅ Carga completada")


def test_queries():
    """Prueba algunas consultas de ejemplo"""
    conn = connect_db()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    print("\n🔍 CONSULTAS DE PRUEBA:")
    print("=" * 50)
    
    # 1. Pinturas por estilo
    print("\n1. Pinturas de estilo 'Urbano':")
    cursor.execute("SELECT archivo, confianza FROM pinturas WHERE estilo_principal = 'Urbano' LIMIT 5")
    for row in cursor.fetchall():
        print(f"   - {row[0]} ({row[1]*100:.1f}%)")
    
    # 2. Pinturas con alta confianza
    print("\n2. Pinturas con confianza > 12%:")
    cursor.execute("SELECT archivo, estilo_principal, confianza FROM pinturas WHERE confianza > 0.12 ORDER BY confianza DESC LIMIT 5")
    for row in cursor.fetchall():
        print(f"   - {row[0]}: {row[1]} ({row[2]*100:.1f}%)")
    
    # 3. Contar por estilo
    print("\n3. Total por estilo:")
    cursor.execute("SELECT * FROM resumen_estilos")
    for row in cursor.fetchall():
        print(f"   - {row[0]}: {row[1]}")
    
    cursor.close()
    conn.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_queries()
    else:
        load_catalog()
        print("\n" + "=" * 50)
        print("💡 Para probar consultas ejecuta: python load_to_db.py test")
