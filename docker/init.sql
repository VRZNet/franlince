-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Paintings table
CREATE TABLE IF NOT EXISTS pinturas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    archivo VARCHAR(255) NOT NULL,
    ruta VARCHAR(500),
    imagen BYTEA,
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

-- Indexes for fast searches
CREATE INDEX IF NOT EXISTS idx_pinturas_estilo ON pinturas(estilo_principal);
CREATE INDEX IF NOT EXISTS idx_pinturas_archivo ON pinturas(archivo);

-- Vector similarity search index (IVFFlat)
-- Created after having data with: CREATE INDEX ON pinturas USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

-- Function to automatically update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for updated_at
CREATE TRIGGER update_pinturas_updated_at
    BEFORE UPDATE ON pinturas
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Style summary view
CREATE OR REPLACE VIEW resumen_estilos AS
SELECT
    estilo_principal,
    COUNT(*) as cantidad,
    ROUND(AVG(confianza)::numeric, 3) as confianza_promedio
FROM pinturas
GROUP BY estilo_principal
ORDER BY cantidad DESC;
