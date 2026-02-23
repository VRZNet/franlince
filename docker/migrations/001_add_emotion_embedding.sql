-- Migration: Add emotion embedding and emotion classification fields
-- Run this migration to add emotional search capabilities

-- Add emotional embedding column (512 dimensions like visual embedding)
ALTER TABLE pinturas
ADD COLUMN IF NOT EXISTS embedding_emocional vector(512);

-- Add emotion classification fields
ALTER TABLE pinturas
ADD COLUMN IF NOT EXISTS emocion_principal VARCHAR(50);

ALTER TABLE pinturas
ADD COLUMN IF NOT EXISTS emocion_confianza FLOAT;

ALTER TABLE pinturas
ADD COLUMN IF NOT EXISTS emocion_2 VARCHAR(50);

ALTER TABLE pinturas
ADD COLUMN IF NOT EXISTS emocion_2_confianza FLOAT;

ALTER TABLE pinturas
ADD COLUMN IF NOT EXISTS emocion_3 VARCHAR(50);

ALTER TABLE pinturas
ADD COLUMN IF NOT EXISTS emocion_3_confianza FLOAT;

ALTER TABLE pinturas
ADD COLUMN IF NOT EXISTS todas_emociones JSONB;

-- Create index for emotional embedding search
-- Note: Run this after populating emotional embeddings for better performance
-- CREATE INDEX IF NOT EXISTS idx_pinturas_embedding_emocional
-- ON pinturas USING ivfflat (embedding_emocional vector_cosine_ops) WITH (lists = 10);

-- Create index for emotion filtering
CREATE INDEX IF NOT EXISTS idx_pinturas_emocion ON pinturas(emocion_principal);

-- Update the style summary view to include emotions
CREATE OR REPLACE VIEW resumen_emociones AS
SELECT
    emocion_principal,
    COUNT(*) as cantidad,
    ROUND(AVG(emocion_confianza)::numeric, 3) as confianza_promedio
FROM pinturas
WHERE emocion_principal IS NOT NULL
GROUP BY emocion_principal
ORDER BY cantidad DESC;

-- Combined summary view
CREATE OR REPLACE VIEW resumen_completo AS
SELECT
    estilo_principal,
    emocion_principal,
    COUNT(*) as cantidad,
    ROUND(AVG(confianza)::numeric, 3) as confianza_estilo_promedio,
    ROUND(AVG(emocion_confianza)::numeric, 3) as confianza_emocion_promedio
FROM pinturas
GROUP BY estilo_principal, emocion_principal
ORDER BY cantidad DESC;
