-- Habilitar extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabla de pinturas
CREATE TABLE IF NOT EXISTS pinturas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    archivo VARCHAR(255) NOT NULL,
    ruta VARCHAR(500),
    estilo_principal VARCHAR(50) NOT NULL,
    confianza FLOAT NOT NULL,
    estilo_2 VARCHAR(50),
    confianza_2 FLOAT,
    estilo_3 VARCHAR(50),
    confianza_3 FLOAT,
    todos_estilos JSONB,
    embedding vector(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices para búsquedas rápidas
CREATE INDEX IF NOT EXISTS idx_pinturas_estilo ON pinturas(estilo_principal);
CREATE INDEX IF NOT EXISTS idx_pinturas_archivo ON pinturas(archivo);

-- Índice para búsqueda por similitud vectorial (IVFFlat)
-- Se crea después de tener datos con: CREATE INDEX ON pinturas USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

-- Función para actualizar updated_at automáticamente
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger para updated_at
CREATE TRIGGER update_pinturas_updated_at
    BEFORE UPDATE ON pinturas
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Vista de resumen por estilo
CREATE OR REPLACE VIEW resumen_estilos AS
SELECT 
    estilo_principal,
    COUNT(*) as cantidad,
    ROUND(AVG(confianza)::numeric, 3) as confianza_promedio
FROM pinturas
GROUP BY estilo_principal
ORDER BY cantidad DESC;
