"""
Constants for Franlince API.
Contains style categories and other constant values.
"""

from typing import Dict, List, Set

# Style categories with multiple prompts for CLIP classification
STYLE_CATEGORIES: Dict[str, List[str]] = {
    "Paisaje": [
        "a painting of mountains and valleys",
        "a painting of countryside with trees and fields",
        "a painting of sunset or sunrise over hills",
        "a landscape painting with sky and nature",
        "a painting of forests and rivers in nature"
    ],
    "Marino": [
        "a painting of ocean waves and sea",
        "a painting of boats and ships on water",
        "a painting of beach with sand and waves",
        "a seascape painting with blue water",
        "a painting of lighthouse by the sea"
    ],
    "Abstracto": [
        "an abstract painting with swirling colors and shapes",
        "a modern abstract painting with gold and blue textures",
        "an abstract art with fluid flowing patterns",
        "a contemporary abstract painting with metallic colors",
        "an abstract painting with geometric shapes and bold colors"
    ],
    "Retrato": [
        "a portrait painting of a person face",
        "an artistic portrait of a woman with decorative elements",
        "a stylized portrait with gold accents",
        "a portrait painting showing human face and shoulders",
        "a decorative portrait with artistic background"
    ],
    "Naturaleza Muerta": [
        "a still life painting with fruits on table",
        "a still life painting with vases and flowers",
        "a painting of wine bottles and glasses",
        "a still life with food and kitchen objects",
        "a painting of objects arranged on a surface"
    ],
    "Urbano": [
        "a street art style painting with graffiti",
        "a pop art painting with bold colors and icons",
        "a painting of classical sculpture with modern graffiti background",
        "an urban art painting mixing classical and street style",
        "a painting with hearts and graffiti street art style"
    ],
    "Floral": [
        "a painting of roses and flowers",
        "a floral painting with colorful blooms",
        "a painting of sunflowers",
        "a decorative painting with flower arrangements",
        "a painting featuring flowers as main subject"
    ],
    "Fauna": [
        "a painting of wild animals",
        "a painting of elephants or lions",
        "a painting of horses running",
        "a painting of birds in nature",
        "an animal portrait painting"
    ],
    "Religioso": [
        "a religious painting with Jesus Christ",
        "a painting of Virgin Mary",
        "a painting with angels and saints",
        "a biblical scene painting",
        "a painting of the last supper or crucifixion"
    ]
}

# Allowed image types for upload
ALLOWED_IMAGE_TYPES: List[str] = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp"
]

# Image file extensions
IMAGE_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"
}

# Media type mapping for serving images
MEDIA_TYPES: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp"
}

# Classification temperature for softmax
CLASSIFICATION_TEMPERATURE: float = 0.5

# Minimum similarity threshold for semantic search
# CLIP text-image similarities are typically in 0.20-0.35 range for good matches
# Using 0.28 to filter out irrelevant results while keeping good matches
MIN_SIMILARITY_THRESHOLD: float = 0.28

# Minimum similarity for visual content in hybrid search
# Increased from 0.22 to 0.28 to prevent irrelevant results
# (e.g., searching "perros" shouldn't return "woman with flowers")
MIN_VISUAL_SIMILARITY: float = 0.28

# Minimum similarity for emotional content in hybrid search
MIN_EMOTION_SIMILARITY: float = 0.22

# Hard minimum similarity - backend enforced floor that cannot be bypassed
# by frontend query parameters. Prevents returning completely irrelevant results.
HARD_MIN_SIMILARITY: float = 0.26

# =============================================================================
# EMOTION CLASSIFICATION CONFIGURATION
# =============================================================================
# Aggregation method for emotion classification:
#   - "max": Use only the highest scoring prompt (most discriminative)
#   - "mean": Average all prompts (NOT recommended - dilutes scores)
#   - "top_k_mean": Average top K prompts
#   - "top_k_max": Weighted combination of top K prompts (RECOMMENDED)
EMOTION_AGGREGATION_METHOD: str = "top_k_max"

# Number of top prompts to consider when using top_k methods
EMOTION_TOP_K: int = 3

# Softmax temperature for emotion classification
# Lower = more discriminative (bigger difference between top scores)
# Higher = softer distribution (scores closer together)
EMOTION_CLASSIFICATION_TEMPERATURE: float = 0.4  # Default was 0.75

# =============================================================================
# EMOTION CATEGORIES
# =============================================================================
# Emotion categories with multiple prompts for CLIP classification
EMOTION_CATEGORIES: Dict[str, List[str]] = {
   "alegria": [
        # Elementos visuales concretos - fiestas, celebraciones
        "painting of people dancing, laughing and celebrating together",
        "artwork depicting festivals, carnivals or street celebrations with vivid colors",
        "painting of children playing outdoors in sunlight and open fields",
        
        # Paleta y estilo - CLIP detecta colores
        "bright and colorful Impressionist painting of leisure and joyful gatherings",
        "painting with warm yellows, vibrant reds and energetic brushstrokes conveying happiness",
        
        # Gestos y expresiones corporales
        "painting of figures with arms raised, spinning or embracing in pure joy",
        "artwork capturing laughter, wide smiles and spontaneous human delight",
        
        # Escenas cotidianas de alegría
        "artwork showing musicians playing and crowds enjoying music and dance",
        
        # Referencias clásicas que CLIP conoce bien
        "Renoir or Matisse style painting of joyful social gatherings and leisure",
        
        # Naturaleza como alegría
        "painting of blooming flowers, sunny landscapes and birds in flight evoking happiness",
        "artwork of children and animals playing freely in nature on a bright day",
    ],
   "paz": [
        # Escenas naturales tranquilas - lo más visual posible
        "painting of a still lake reflecting mountains at dawn with no wind",
        "artwork of a quiet forest path with soft light filtering through trees",
        "painting of a calm ocean at sunset with gentle waves and empty horizon",
        
        # Espacios íntimos de quietud
        "artwork of a person sitting alone by a window in soft morning light",
        "painting of an empty garden, courtyard or chapel in silence and stillness",
        "artwork depicting a sleeping figure in a warm, softly lit room",
        
        # Paleta específica que CLIP asocia a paz
        "painting in soft blues, muted greens and pale golds evoking calm and serenity",
        "artwork with minimal composition, open spaces and gentle diffused light",
        
        # Referencias estilísticas fuertes
        "Japanese ink painting of mountains, mist and solitary figures in nature",
        "Corot or Constable landscape painting of quiet countryside in soft morning light",
        
        # Paz entre personas
        "painting of two figures sitting together in comfortable silence",
        "artwork depicting a small village at dusk with warm lights and no movement",
        
        # Abstracto pero visual
        "painting where horizontal lines, open sky and vast water suggest infinite calm",
        "minimalist painting of fog, mist and soft shapes dissolving into stillness",
    ],
   "libertad": [
        # Elementos visuales icónicos de libertad
        "painting of a bird or flock of birds soaring across a vast open sky",
        "artwork of a lone figure standing on a cliff edge facing an infinite horizon",
        "painting of wild horses running free across open plains and grasslands",
        
        # Acción y movimiento - libertad como gesto físico
        "painting of a person with arms wide open facing the wind and open landscape",
        "artwork of a figure running barefoot through fields, hair loose in the wind",
        "painting of a boat or sailboat alone in a vast open sea under big sky",
        
        # Paleta y composición que CLIP asocia a libertad
        "painting dominated by vast open sky, wide horizons and small human figures",
        "artwork with expansive blue skies, soaring birds and golden open landscapes",
        
        # Referencias históricas y simbólicas fuertes
        "Delacroix style painting of rebellion, triumph and liberation with dramatic energy",
        "Romantic painting of solitary figure in wild untamed nature feeling boundless freedom",
        
        # Libertad interior / espiritual
        "painting of chains broken, doors open or cages empty symbolizing liberation",
        "artwork of a figure emerging from darkness into vast open light and space",
        
        # Naturaleza salvaje como libertad
        "painting of untamed wilderness, eagles, open prairies and uncharted landscapes",
        "artwork of wind, open roads, endless skies and the feeling of going anywhere",
    ],
    "energia": [
        # Movimiento físico visible - lo que CLIP detecta
        "painting of figures in explosive athletic motion, sprinting or jumping mid-air",
        "artwork of a storm, lightning strike and turbulent skies with raw power",
        "painting of crashing waves, rushing waterfalls and violent natural forces",
        
        # Pincelada y técnica como energía
        "expressionist painting with thick aggressive brushstrokes and chaotic composition",
        "artwork with swirling lines, fragmented shapes and intense color contrasts",
        "Van Gogh style painting with frenzied spiral brushwork and electric atmosphere",
        
        # Paleta que CLIP asocia a energía
        "painting dominated by electric yellows, fierce reds and sharp contrasting blacks",
        "artwork with high contrast colors, sharp diagonals and unstable dynamic composition",
        
        # Escenas de acción humana
        "painting of a crowd in motion, dancers spinning or athletes in fierce competition",
        "artwork depicting fire, sparks, explosions or bursts of light and kinetic force",
        
        # Referencias estilísticas poderosas
        "Futurist painting capturing speed, machines and the raw energy of modern life",
        "Abstract Expressionist artwork of bold gestural marks conveying explosive inner force",
        
        # Naturaleza como energía pura
        "painting of a thunderstorm over the ocean with dramatic dark clouds and lightning",
        "artwork of wildfire, volcanic eruption or natural forces unleashed with intensity",

        # Personajes en acción dinámica
        "cartoon or animated character jumping running or punching with dynamic action pose and bold colors",
        "illustration of a heroic figure mid-leap with fist forward in explosive energetic motion",
    ],
    "romanticismo": [
        # Gestos físicos concretos de amor
        "painting of two figures embracing tenderly in soft warm light",
        "artwork of lovers holding hands walking through a sunlit garden or park",
        "painting of a couple sharing a quiet intimate moment by candlelight",
        
        # Escenas románticas clásicas
        "artwork of a couple in a rowboat on a calm lake surrounded by weeping willows",
        "painting of a woman reading a love letter by a window in golden afternoon light",
        "artwork depicting a stolen glance, a gentle touch or a first kiss between lovers",
        
        # Paleta y atmósfera que CLIP asocia a romance
        "painting in warm rose golds, deep crimsons and soft candlelight atmospheres",
        "artwork with soft focus, warm glowing light and intimate close compositions",
        
        # Referencias estilísticas fuertes
        "Renoir style painting of couples dancing, laughing and flirting in dappled light",
        "Pre-Raphaelite painting of beautiful romantic figures in lush decorative settings",
        
        # Romanticismo en la naturaleza
        "painting of two silhouettes against a dramatic sunset or moonlit landscape",
        "artwork of flowers, roses and garden settings as symbols of romantic love",
        
        # Intimidad emocional
        "painting capturing the vulnerability and tenderness of two people deeply in love",
        "artwork of a sleeping beloved, a whispered secret or a lingering farewell embrace",
    ],
    "nostalgia": [
        # Escenas de infancia y memoria
        "painting of children playing in summer streets of a small town long ago",
        "artwork of an old family home, porch swing and overgrown garden at dusk",
        "painting of a grandmother's kitchen with worn furniture and fading wallpaper",
        
        # Objetos físicos que evocan el pasado
        "artwork of old photographs, faded letters, pocket watches and worn childhood toys",
        "painting of dusty bookshelves, vintage radios and objects from a forgotten era",
        "artwork depicting an abandoned schoolroom, empty playground or closed general store",
        
        # Luz y atmósfera nostálgica
        "painting bathed in warm sepia tones and golden late afternoon light of distant memory",
        "artwork with soft blurred edges, faded colors and dreamlike quality of old photographs",
        
        # Referencias estilísticas que CLIP asocia a nostalgia
        "Norman Rockwell style painting of idealized small town American life and simpler times",
        "Edward Hopper painting of empty diners, lonely streets and quiet melancholy spaces",
        
        # Naturaleza y tiempo que pasa
        "painting of an old tree in a field where children once played, now empty and silent",
        "artwork of a summer evening fading to dusk in a place that no longer exists as remembered",
        
        # Figuras mirando hacia atrás
        "painting of an elderly person holding an old photograph lost in distant memory",
        "artwork of a lone figure standing before a childhood home now changed beyond recognition",
    ],
   "misterio": [
        # Escenas visuales oscuras y enigmáticas
        "painting of a dimly lit corridor, staircase or doorway leading into unknown darkness",
        "artwork of a hooded or masked figure standing alone in shadow and fog",
        "painting of an abandoned mansion, ruined castle or forgotten room with hidden secrets",
        
        # Naturaleza misteriosa
        "artwork of a dark forest path disappearing into dense fog and shadow",
        "painting of a moonlit lake with strange reflections and figures barely visible",
        "artwork of storm clouds, ravens and an eerie landscape under pale moonlight",
        
        # Símbolos y objetos enigmáticos
        "painting filled with cryptic symbols, ancient maps, locked boxes and hidden clues",
        "artwork of a candle casting long shadows over mysterious objects on a table",
        "painting of mirrors, reflections and shadows that reveal something unexpected",
        
        # Paleta y composición que CLIP asocia a misterio
        "painting dominated by deep blacks, midnight blues and single dramatic light sources",
        "artwork with strong chiaroscuro, hidden faces and figures emerging from darkness",
        
        # Referencias estilísticas poderosas
        "Rembrandt style painting with dramatic spotlight emerging from deep shadow and darkness",
        "Symbolist painting of dreamlike figures, veiled women and ambiguous dark narratives",
        
        # Figura humana y misterio
        "painting of a solitary figure seen from behind contemplating something unseen",
        "artwork of a stranger's face half hidden in shadow with unreadable expression",
    ],
    "abundancia": [
        # Escenas visuales de cosecha y plenitud
        "painting of overflowing baskets of fruits, vegetables and flowers at harvest time",
        "artwork of a lavish banquet table covered with food, wine and overflowing platters",
        "painting of a cornucopia spilling fruits, grapes, wheat and flowers in abundance",

        # Naturaleza en plenitud
        "artwork of an orchard heavy with ripe fruit under golden summer sunlight",
        "painting of fertile fields, golden wheat harvest and farmers celebrating plenty",
        "artwork of a market stall overflowing with colorful fruits, spices and fresh produce",

        # Riqueza material y monetaria
        "painting of piles of gold coins, stacks of money and overflowing treasure chests",
        "artwork of a figure surrounded by wealth, cash, gold bars and precious jewels",
        "painting of a vault or room filled with money, gold and material riches in excess",

        # Interiores ricos y opulentos
        "painting of a wealthy interior with silk fabrics, gold objects and lavish decorations",
        "artwork of a Dutch Golden Age still life with exotic fruits, fine glassware and flowers",

        # Paleta que CLIP asocia a abundancia
        "painting dominated by rich golds, deep greens and warm earthy tones of harvest season",
        "artwork with dense layered composition leaving no empty space, filled with life and color",

        # Referencias estilísticas poderosas
        "Flemish Baroque still life painting of exotic foods, game, flowers and precious objects",
        "Rubens style painting of full figured figures surrounded by fruits and natural bounty",

        # Celebración, lujo y prosperidad
        "painting of people sharing food at a large communal feast with joy and generosity",
        "artwork depicting luxury, opulence and material prosperity with gold and rich textures",
    ],
    "ternura": [
        # Ancla semántica fuerte - SIEMPRE dos sujetos en contacto directo
        "painting of tenderness expressed between two people through gentle physical contact and soft gaze",
        "artwork where warmth comes exclusively from human connection between figures not solitary labor",
        "painting of tender care between two specific figures not a solitary working figure alone",
        
        # Vínculos físicos más universales de ternura
        "painting of a mother cradling a newborn baby in soft warm light",
        "artwork of a parent kissing a sleeping child on the forehead gently",
        "painting of an elderly couple holding hands with quiet lifelong devotion",
        
        # Gestos pequeños y delicados entre dos personas
        "artwork of gentle hands braiding a child's hair with care and patience",
        "painting of a person tenderly caring for a sick or sleeping loved one",
        "artwork of a child pressing their face against a soft animal or beloved toy",
        
        # Animales y ternura
        "painting of a mother animal nursing her young in a warm sheltered space",
        "artwork of a child sleeping curled up with a dog or cat in afternoon light",
        "painting of a bird feeding her chicks in a nest with protective gentleness",
        
        # Paleta - blindada contra trabajo solitario y flores como carga
        "painting in soft pinks and warm creams where two or more figures share gentle intimate connection",
        "artwork with blurred soft edges and warm candlelight glow shared between two close figures",
        
        # Referencias estilísticas
        "Mary Cassatt Impressionist painting of mothers and children in intimate domestic moments",
        "Raphael Madonna painting of mother and child with divine tenderness and soft grace",
        
        # Momentos cotidianos de cuidado entre personas
        "painting of a grandmother reading to a small child tucked under a warm blanket",
        "artwork of two siblings sharing a whispered secret with gentle smiles and closeness",
    ],
    "fuerza": [
        # Fuerza física visible y concreta
        "painting of a powerful athletic figure with muscular form in peak physical strength",
        "artwork of a warrior, gladiator or soldier standing firm against overwhelming odds",
        "painting of a blacksmith, stonemason or laborer wielding tools with raw physical power",
        
        # Fuerza en la naturaleza como espejo
        "artwork of a lone oak tree standing unbroken against a violent storm and strong winds",
        "painting of a bull, lion or bear as symbols of raw untamed natural strength and power",
        "artwork of a mountain, cliff face or massive boulder immovable against crashing waves",
        
        # Fuerza interior y emocional
        "painting of a solitary figure standing tall against darkness with quiet unshakeable resolve",
        "artwork of a person rising from the ground battered but refusing to stay down",
        "painting of a clenched fist, set jaw and determined eyes facing an impossible challenge",
        
        # Paleta y composición que CLIP asocia a fuerza
        "painting with bold dark contours, strong diagonal lines and dramatic upward compositions",
        "artwork dominated by deep blacks, fierce reds and sharp contrasts suggesting raw power",
        
        # Referencias estilísticas poderosas
        "Michelangelo style painting of heroic muscular figures straining with divine strength",
        "Goya painting of a lone figure defiant and unbowed in the face of overwhelming force",
        
        # Fuerza colectiva
        "painting of a crowd of workers or soldiers united and advancing with collective strength",
        "artwork of figures lifting, pulling or pushing together with shared unstoppable force",
    ],
    "aventura": [
        # Exploración física concreta
        "painting of explorers or adventurers navigating uncharted wilderness and dense jungle",
        "artwork of a sailing ship battling massive ocean waves under dramatic stormy skies",
        "painting of mountain climbers ascending a treacherous snow covered peak at dawn",
        
        # Caminos y horizontes desconocidos
        "artwork of a lone traveler on horseback disappearing into a vast unknown landscape",
        "painting of an ancient map, compass and explorer's tools spread on a worn wooden table",
        "artwork of a winding road or river trail disappearing into uncharted wilderness ahead",
        
        # Momentos de descubrimiento
        "painting of an explorer stepping ashore on an undiscovered coastline for the first time",
            "artwork of archaeologists uncovering ancient ruins in a remote jungle expedition",
        "painting capturing the exact moment of arriving somewhere completely new and unknown",
        
        # Naturaleza salvaje e inhóspita
        "artwork of vast untamed wilderness, towering cliffs and roaring rivers defying passage",
        "painting of a campfire at night under a vast starlit sky in remote wild territory",
        "artwork of wild animals, dense forests and landscapes that have never seen human feet",
        
        # Referencias estilísticas poderosas
        "Frederic Church Romantic landscape painting of epic wilderness and geographical discovery",
        "19th century adventure illustration of explorers, maps and exotic faraway destinations",
        
        # Acción y peligro
        "painting of a figure leaping across a chasm, crossing a rope bridge or scaling a cliff",
        "artwork of a small human figure dwarfed by enormous untamed natural forces and landscape",

        # Personajes animados en aventura
        "cartoon hero character on an epic quest running through colorful worlds collecting treasures",
        "illustration of an animated adventurer jumping over obstacles in a fantastical vibrant world",
    ],
   "esperanza": [
        # Luz como metáfora visual más universal de esperanza
        "painting of a single ray of sunlight breaking through dark storm clouds over landscape",
        "artwork of dawn light slowly illuminating a dark horizon after a long night",
        "painting of a lighthouse beam cutting through fog and darkness guiding ships to safety",
        
        # Naturaleza renaciendo
        "artwork of the first flowers pushing through snow at the end of a long winter",
        "painting of a seedling sprouting from cracked dry earth reaching toward sunlight",
        "artwork of a rainbow appearing over a landscape still wet from recent storm",
        
        # Figuras humanas mirando hacia adelante
        "painting of a figure standing at a threshold, open door or hilltop facing a bright horizon",
        "artwork of a mother holding a newborn baby looking out a window toward the future",
        "painting of a child reaching upward toward light with open hands and upturned face",
        
        # Paleta específica que CLIP asocia a esperanza
        "painting transitioning from dark lower tones to luminous warm light in upper composition",
        "artwork dominated by pale golds, soft blues and the gentle light of early morning",
        
        # Referencias estilísticas poderosas
        "George Frederic Watts allegorical painting of a blindfolded figure clinging to hope",
        "Turner painting of radiant light dissolving darkness over a vast luminous landscape",
        
        # Gestos y símbolos universales
        "painting of hands extended upward, seeds being planted or a bird released into open sky",
        "artwork of a small boat navigating calm waters toward a glowing distant horizon at dawn",
    ],
   "rebeldia": [
        # Confrontación física directa
        "painting of a lone figure standing defiantly before an armed crowd or authority",
        "artwork of protesters marching with raised fists and banners through city streets",
        "painting of a person tearing down a statue, flag or symbol of oppressive power",
        
        # Gestos de desafío concretos
        "artwork of a figure refusing to kneel, bow or submit before overwhelming authority",
        "painting of a young person staring down soldiers, police or figures of power unflinching",
        "artwork of broken chains, smashed barriers and torn uniforms of conformity",
        
        # Naturaleza como rebeldía
        "painting of a single wildflower growing through cracks in concrete and stone walls",
        "artwork of a lone wolf or black sheep separated defiantly from the obedient herd",
        "painting of lightning striking an established institution or crumbling monument of power",
        
        # Paleta y composición que CLIP asocia a rebeldía
        "painting with aggressive slashing brushstrokes, clashing colors and unstable compositions",
        "artwork dominated by stark blacks, fierce reds and sharp angular forms suggesting revolt",
        
        # Referencias estilísticas poderosas
        "Delacroix Liberty Leading the People style painting of revolutionary masses rising up",
        "Goya painting of civilian defiance and resistance against brutal occupying military force",
        
        # Rebeldía cultural y artística
        "painting of an artist or outsider figure rejecting the established order with bold gesture",
        "artwork capturing the raw energy of punk, counterculture or youth revolt against convention",
    ],
   "sabiduria": [
        # Figuras sabias concretas y reconocibles
        "painting of an elderly sage or philosopher in deep contemplation surrounded by books",
        "artwork of a white bearded elder sharing wisdom with attentive younger figures",
        "painting of a solitary scholar studying ancient manuscripts by candlelight at night",
        
        # Objetos y símbolos visuales de sabiduría
        "artwork of ancient scrolls, hourglasses, celestial globes and instruments of learning",
        "painting of an owl perched on old books beside a burning candle in a scholar's study",
        "artwork of a weathered tree with deep roots as metaphor for accumulated wisdom and age",
        
        # Escenas de transmisión del conocimiento
        "painting of a mentor and student in quiet focused dialogue beside a window",
        "artwork of an elder teaching young children gathered around listening with open attention",
        "painting of Socrates, Confucius or ancient philosophers in animated intellectual discourse",
        
        # Paleta y atmósfera que CLIP asocia a sabiduría
        "painting in warm amber tones, candlelight and deep shadows of a scholar's intimate space",
        "artwork with aged textures, worn surfaces and the warm patina of accumulated years",
        
        # Referencias estilísticas poderosas
        "Rembrandt portrait of an elderly scholar with deeply lined face radiating quiet wisdom",
        "Raphael School of Athens style painting of great philosophers in intellectual gathering",
        
        # Sabiduría interior y contemplativa
        "painting of a figure in meditative stillness having arrived at deep inner understanding",
        "artwork of an ancient face with eyes that have witnessed centuries of human experience",
    ],
   "union": [
        # Contacto físico directo - lo más visual posible
        "painting of many hands joined together in a circle from above",
        "artwork of a large family embracing together in a warm crowded group hug",
        "painting of workers, soldiers or teammates with arms around each other's shoulders",
        
        # Escenas comunitarias concretas
        "artwork of a village gathering around a shared meal at long communal tables outdoors",
        "painting of a congregation, choir or crowd singing together with one shared voice",
        "artwork of neighbors building a barn, house or structure together with collective effort",
        
        # Vínculos familiares específicos
        "painting of three or four generations of a family gathered in a warm domestic interior",
        "artwork of siblings holding hands walking together down a road or through a field",
        "painting of a circle of children dancing together hand in hand in an open field",
        
        # Unión en la adversidad
        "artwork of people sheltering together under one roof during a storm sharing warmth",
        "painting of a community standing together facing the same direction with shared purpose",
        "artwork of rescue workers, neighbors or strangers helping each other after disaster",
        
        # Paleta y composición que CLIP asocia a unión
        "painting with circular or convergent compositions where all figures lean toward a center",
        "artwork with warm firelight illuminating multiple faces gathered closely together",
        
        # Referencias estilísticas poderosas
        "Diego Rivera mural style painting of workers and communities united in collective strength",
        "Bruegel style painting of peasant communities celebrating together in joyful gatherings",
        
        # Unión simbólica
        "painting of intertwined roots, branches or rivers merging as metaphor for human bonds",
        "artwork of diverse figures from different backgrounds standing shoulder to shoulder as one",
    ],
    "lealtad": [
        # Vínculos humanos concretos de lealtad
        "painting of a soldier standing guard over fallen comrades refusing to abandon them",
        "artwork of a faithful servant or squire beside their lord through battle and hardship",
        "painting of two old friends reuniting after years apart with unbroken mutual trust",
        
        # El perro como símbolo visual más universal de lealtad
        "artwork of a loyal dog sitting vigil beside its owner's grave waiting faithfully",
        "painting of a dog greeting its returning owner with unconditional devoted joy",
        "artwork of a faithful hound accompanying its master through storm and wilderness",
        
        # Lealtad en la adversidad
        "painting of a figure standing by a disgraced or fallen friend when all others have left",
        "artwork of two soldiers or companions carrying each other across a difficult battlefield",
        "painting of a person refusing to betray a friend despite pressure and personal cost",
        
        # Gestos físicos de lealtad
        "artwork of a sworn oath with hands clasped or raised before witnesses with solemn gravity",
        "painting of figures kneeling in pledge of allegiance with hands over hearts",
        "artwork of a handshake or clasped hands between two figures sealing an unbreakable bond",
        
        # Paleta y atmósfera que CLIP asocia a lealtad
        "painting in steady deep blues, warm golds and the quiet dignity of unwavering commitment",
        "artwork with solid grounded compositions and figures standing firm side by side",
        
        # Referencias estilísticas poderosas
        "painting in the style of Landseer depicting noble dogs and their devoted faithful bonds",
        "Jacques-Louis David Neoclassical painting of solemn oath and brotherhood under shared cause",
        
        # Lealtad a lo largo del tiempo
        "painting of an elderly couple who have kept faith with each other through an entire lifetime",
        "artwork of a parent waiting at a window for a child who has been long away from home",
    ],
    "humor": [
        # Escenas cómicas visuales concretas
        "painting of a jester or clown caught in an absurd comedic situation with exaggerated expression",
        "artwork of figures in slapstick mishap, tripping stumbling or caught in ridiculous predicament",
        "painting of animals behaving like humans in amusing domestic or social situations",
        
        # Expresiones y gestos cómicos
        "artwork of exaggerated facial expressions, bulging eyes and comically oversized features",
        "painting of characters with absurdly exaggerated proportions in a satirical comic scene",
        "artwork of a pompous dignitary or authority figure caught in undignified embarrassing moment",
        
        # Humor social y sátira visual
        "painting of wealthy aristocrats or pompous figures being mocked by clever common people",
        "artwork satirizing the pretensions of the powerful with sharp wit and visual irony",
        "painting of a chaotic market, tavern brawl or carnival scene full of comic mischief",
        
        # Humor absurdo y whimsical
        "artwork of impossible whimsical scenes where gravity rules and logic is gleefully abandoned",
        "painting of anthropomorphic animals in human clothes engaged in serious absurd activities",
        "artwork of a dreamlike scene where everyday objects behave in hilariously unexpected ways",
        
        # Paleta y estilo que CLIP asocia a humor
        "painting with bright exaggerated colors, bouncy rounded forms and cheerful chaotic energy",
        "artwork with cartoonish exaggeration, bold outlines and gleefully distorted reality",
        
        # Referencias estilísticas poderosas
        "Hogarth satirical painting of English society caught in absurd comic moral situations",
        "Bruegel painting of peasant festivities with rowdy slapstick humor and comic village chaos",
        
        # Humor tierno y ligero
        "painting of children playing a prank with mischievous grins and barely suppressed laughter",
        "artwork of a dog or cat in a hilariously undignified position with complete oblivious dignity",
    ],
    "dignidad": [
        # Postura y presencia física - lo más visual de dignidad
        "painting of a person standing tall with composed upright posture and quiet commanding presence",
        "artwork of an elder or leader with calm unshakeable bearing despite surrounding adversity",
        "painting of a figure who has suffered much but carries themselves with unbroken inner nobility",
        
        # Dignidad en contextos de adversidad
        "artwork of a formerly enslaved person standing free with profound quiet self possession",
        "painting of an indigenous elder in ceremonial dress with proud dignified undefeated bearing",
        "artwork of a poor laborer or servant carrying themselves with the same grace as any king",
        
        # El rostro como expresión de dignidad
        "painting of a weathered face with calm steady eyes that refuse to be diminished or humiliated",
        "artwork of an elderly woman with deeply lined face radiating quiet unassailable inner worth",
        "painting of a person receiving humiliation but maintaining complete composure and self respect",
        
        # Dignidad colectiva
        "artwork of marginalized people walking together with heads held high and collective pride",
        "painting of workers or common people portrayed with the same gravitas as nobility or royalty",
        
        # Paleta y composición que CLIP asocia a dignidad
        "painting with centered stable vertical compositions and figures that occupy space with authority",
        "artwork in deep measured tones of navy, burgundy and gold suggesting quiet noble bearing",
        
        # Referencias estilísticas poderosas
        "Velázquez portrait style painting of common people rendered with aristocratic dignity and respect",
        "painting in the tradition of Käthe Kollwitz showing working class dignity and quiet human nobility",
        
        # Dignidad como silencio y composure
        "painting of a figure alone in difficult circumstances maintaining complete stillness and grace",
        "artwork capturing the moment someone chooses dignity over rage when faced with injustice",
    ],
    "esfuerzo": [
        # Elementos visuales físicos
        "painting of workers, laborers or athletes in intense physical exertion",
        "artwork depicting sweat, strain and muscular effort in human figures",
        "painting of hands worn by work, tools and the marks of hard labor",
        
        # Carga física - ancla principal para Rivera
        "painting of a small crouching figure completely overwhelmed by an enormous load they carry",
        "artwork of a lone worker bent under the weight of goods flowers or produce strapped to their back",
        "painting of a kneeling figure straining under a burden larger than their entire body",
        "artwork of bare feet and calloused hands of a worker crushed under the physical weight of daily labor",
        "painting of a figure whose entire body is consumed and hidden by the massive load they carry alone",
        
        # Narrativa del esfuerzo
        "painting of a person climbing, pushing or carrying a heavy burden uphill",
        "artwork showing exhausted but determined figures pressing forward",
        "painting capturing the moment before giving up but choosing to continue",
        
        # Contexto histórico / clásico
        "Social realism painting of workers and laborers with dignity and strength",
        "Diego Rivera style painting of Mexican laborers carrying enormous loads of flowers or produce",
        "classical painting depicting Sisyphus, Hercules or mythological figures of endurance",
        
        # Abstracto / simbólico
        "painting where rough textures and dark tones convey struggle and resilience",
        "artwork using stormy skies or rough terrain as metaphor for life's challenges",
        
        # Emocional
        "painting of determined human expression, clenched jaw, focused eyes under pressure",
        "artwork capturing quiet sacrifice and the invisible weight of daily persistence",
    ],
    "soledad": [
        # Figura sola en espacios vastos - arquetipo visual más poderoso
        "painting of a tiny solitary figure dwarfed by an enormous empty landscape or seascape",
        "artwork of a lone person sitting at the edge of a cliff watching the horizon at dusk",
        "painting of a single figure walking away down an endless empty road or beach alone",
        
        # Espacios urbanos vacíos y aislamiento moderno
        "artwork of a person sitting alone in an empty diner or cafe late at night under harsh light",
        "painting of a solitary figure in a crowded city street completely unnoticed and invisible",
        "artwork of an empty apartment with one chair by a window overlooking a silent city at night",
        
        # Soledad en la naturaleza
        "painting of a lone tree standing isolated in a vast empty winter field under grey sky",
        "artwork of a small boat alone on a vast dark ocean with no land visible in any direction",
        "painting of a solitary figure in a snowstorm with no shelter and no other soul in sight",
        
        # Objetos que evocan ausencia
        "artwork of a single empty chair, one unused cup or a bed with one side never slept in",
        "painting of a table set for two with one place untouched and a candle burning alone",
        "artwork of a window with one light on in a dark building surrounded by darkness and silence",
        
        # Paleta y composición que CLIP asocia a soledad
        "painting with vast empty negative space overwhelming a single small human presence",
        "artwork in cold blues, grey silvers and muted tones of a world emptied of companionship",
        
        # Referencias estilísticas poderosas
        "Edward Hopper painting of urban isolation and the profound loneliness of modern life",
        "Caspar David Friedrich painting of a solitary figure contemplating infinite lonely landscape",
        
        # Soledad ambigua - entre paz y melancolía
        "painting of a person reading alone by a window in fading afternoon light with quiet resignation",
        "artwork of a figure standing at a window watching rain fall on an empty street below",
    ],
    "caos": [
        # Caos natural - fuerzas físicas descontroladas
        "painting of a massive storm at sea with ships torn apart by enormous crashing waves",
        "artwork of a volcanic eruption engulfing a city in fire ash and unstoppable destruction",
        "painting of a tornado or hurricane destroying everything in its path with savage force",
        
        # Caos humano y batalla
        "artwork of a battlefield in total disorder with figures falling colliding and screaming",
        "painting of a city riot with crowds surging fire burning and everything breaking apart",
        "artwork of a stampede panic or crowd crushing itself in blind terrified confusion",
        
        # Caos visual en composición y técnica
        "painting with violently clashing colors exploding in all directions with no center or order",
        "artwork of fragmented broken forms colliding and dissolving into pure visual disorder",
        "painting where multiple conflicting perspectives and scales destroy any sense of stability",
        
        # Caos mitológico y alegórico
        "artwork depicting the fall of Babylon, Tower of Babel or apocalyptic civilizational collapse",
        "painting of warring gods titans or mythological forces unleashing primordial destruction",
        "artwork of the Last Judgment with masses of figures falling rising and colliding in chaos",
        
        # Paleta y composición que CLIP asocia a caos
        "painting with no horizon line, no stable ground and figures spinning in all directions",
        "artwork dominated by violent diagonal lines clashing angles and destabilizing compositions",
        
        # Referencias estilísticas poderosas
        "Turner painting of a shipwreck or storm with nature completely overwhelming human order",
        "Hieronymus Bosch painting of hellish chaotic scenes teeming with impossible disorder",
        
        # Caos interior y psicológico
        "painting of a fractured shattered mirror reflecting a broken world in infinite disorder",
        "artwork where the canvas itself seems to be tearing dissolving and falling apart violently",
    ],
    "inspiracion": [
        # El acto creativo concreto - único de inspiración
        "painting of an artist standing before a blank canvas in the electric moment before creation",
        "artwork of a composer at a piano seized by a sudden overwhelming musical vision at night",
        "painting of a writer or poet with pen suspended mid-air caught in a flash of pure inspiration",
        
        # El momento del insight creativo
        "artwork of a figure suddenly illuminated by an idea with light visibly emanating from within",
        "painting of Archimedes, Newton or a scientist in the exact moment of sudden discovery",
        "artwork of a sculptor or painter working in total absorbed creative trance and flow state",
        
        # Musas y fuentes de inspiración
        "painting of a classical muse appearing to an artist or poet in a moment of creative revelation",
        "artwork of an artist surrounded by sketches books and objects in a creative studio in chaos",
        "painting of nature directly inspiring an artist sketching outdoors in complete absorbed wonder",
        
        # Paleta específica del momento creativo
        "painting with a single explosive burst of light or color emerging from surrounding darkness",
        "artwork where one luminous area of intense color radiates outward into a darker composition",
        
        # Referencias estilísticas poderosas
        "Vermeer style painting of an artist in their studio absorbed in the act of creation and craft",
        "Romantic painting of a poet or painter seized by divine creative fury and visionary rapture",
        
        # Inspiración como transformación
        "painting of a figure visibly changed by what they have just seen heard or suddenly understood",
        "artwork of hands that have just stopped moving hovering over a creation just brought to life",
    ],
    "familia": [
        # Estructura familiar completa y reconocible
        "painting of parents and children seated together at a dinner table in warm evening light",
        "artwork of a complete family portrait with grandparents parents and children across generations",
        "painting of a father lifting a small child onto his shoulders laughing in an open field",
        
        # Rituales domésticos cotidianos únicos de familia
        "artwork of a mother tucking children into bed in a warm softly lit bedroom at night",
        "painting of a family gathered around a fireplace on a winter evening in comfortable silence",
        "artwork of parents helping children with homework at a kitchen table under warm lamplight",
        
        # Momentos de transición vital familiar
        "painting of parents watching a child take their very first steps with joy and held breath",
        "artwork of a family waving goodbye to a grown child leaving home for the first time",
        "painting of a family gathered around a new baby meeting them for the very first time",
        
        # Trabajo y vida doméstica compartida
        "artwork of a family working together in a garden kitchen or farm with shared daily purpose",
        "painting of siblings playing together unsupervised inventing games in a backyard at dusk",
        
        # Paleta y atmósfera única de familia
        "painting suffused with warm amber interior light suggesting safety shelter and belonging",
        "artwork where every figure leans naturally toward a shared center of warmth and connection",
        
        # Referencias estilísticas poderosas
        "Mary Cassatt Impressionist painting of intimate domestic family moments in warm interior light",
        "Norman Rockwell style painting of an idealized American family in a recognizable daily ritual",
        
        # Familia como refugio
        "painting of a family sheltering together under one roof while a storm rages outside",
        "artwork of a child running toward open parental arms after a long frightening absence",
    ],
    "amor": [
        # Pasión física y erótica - único de amor romántico
        "painting of two lovers in passionate embrace with intense physical and emotional desire",
        "artwork of a kiss so consuming that the world around the figures disappears completely",
        "painting of lovers intertwined in bedsheets in the warm aftermath of passionate intimacy",
        
        # Intensidad emocional única del amor
        "artwork of a figure overwhelmed by love gazing at their beloved with complete vulnerable devotion",
        "painting of two people so absorbed in each other that nothing else in the world exists",
        "artwork of a lover's hands trembling as they reach toward the face of their beloved",
        
        # Amor en la separación y el anhelo
        "painting of a figure standing alone at a window consumed by longing for an absent beloved",
        "artwork of two lovers forced apart reaching toward each other across an impossible distance",
        "painting of a letter pressed to lips eyes closed consumed by love for the absent writer",
        
        # Amor como vulnerabilidad total
        "artwork of a figure completely undone by love stripped of all armor and defense",
        "painting of two people seeing each other truly for the first time with raw open hearts",
        
        # Paleta que CLIP asocia específicamente a amor pasional
        "painting dominated by deep crimsons, passionate reds and warm golden skin tones of intimacy",
        "artwork with close intimate framing where two faces fill the entire canvas in mutual absorption",
        
        # Referencias estilísticas poderosas
        "Klimt painting of lovers in passionate golden embrace dissolving into decorative ecstasy",
        "Francesco Hayez painting of a passionate farewell kiss with intense romantic longing",
        
        # Amor como transformación
        "painting of a figure visibly transformed by love glowing from within with new life and purpose",
        "artwork capturing the exact moment two people realize simultaneously they are deeply in love",
    ],
    "sexualidad": [
        # El cuerpo humano como sujeto central
        "painting of a nude figure in confident sensual repose celebrating the beauty of human form",
        "artwork of a reclining nude in soft light with the quiet confidence of physical self possession",
        "classical painting of Venus or Aphrodite emerging celebrating divine feminine sensual beauty",
        
        # Sensualidad sin necesidad de pareja
        "painting of a single figure in a state of undress with languid sensual self awareness",
        "artwork of a figure caught in a private intimate moment of physical self consciousness",
        "painting of a body partially draped in silk with the tension between concealment and revelation",
        
        # Deseo y atracción
        "artwork of two figures in the charged moment of mutual physical attraction before contact",
        "painting of eyes meeting across a room with unmistakable electric physical desire",
        "artwork of a figure being observed with desire their awareness of being watched electric",
        
        # Desnudo clásico - bien representado en CLIP
        "Titian or Giorgione reclining Venus painting of serene sensual feminine beauty and grace",
        "Ingres Odalisque style painting of luxurious sensual female nude in exotic intimate setting",
        
        # Paleta y atmósfera sensual
        "painting in warm skin tones deep shadows and soft light that caresses the human form",
        "artwork with velvet textures silk fabrics and the warm intimacy of a private bedchamber",
        
        # Sensualidad simbólica
        "painting of ripe fruits flowers and organic forms as symbols of sensual physical abundance",
        "artwork where the texture of paint itself becomes a sensual physical tactile experience",
        
        # Diferenciación de Amor - cuerpo sobre emoción
        "painting where the human body in its physical beauty is the complete and total subject",
        "artwork celebrating physical form texture and sensuality as ends in themselves",
    ],
   "espiritualidad": [
        # Lo sagrado hecho visualmente explícito
        "painting of a figure kneeling in prayer bathed in supernatural golden divine light from above",
        "artwork of a saint or mystic in ecstatic rapture with eyes rolled upward toward heaven",
        "painting of a religious figure with halo aureole or divine light emanating from their body",
        
        # Arquitectura sagrada como contenedor de lo divino
        "artwork of light streaming through stained glass windows of a cathedral onto stone floor below",
        "painting of a solitary figure dwarfed by the immense interior of an ancient sacred temple",
        "artwork of a monk in a monastery cloister in silent prayer surrounded by sacred stillness",
        
        # Ascensión y trascendencia
        "painting of a figure ascending toward light dissolving into pure radiant spiritual energy",
        "artwork of souls rising upward toward divine light leaving the earthly world far below",
        "painting of a visionary figure receiving divine revelation with arms outstretched to heaven",
        
        # Naturaleza como manifestación de lo sagrado
        "artwork of a solitary figure in a vast landscape overwhelmed by the sublime presence of God",
        "painting of sacred mountains, ancient forests or desert landscapes as dwelling places of divine",
        "artwork of a sunrise or shaft of light through clouds as visible manifestation of the sacred",
        
        # Paleta específicamente espiritual
        "painting dominated by ethereal golds, celestial blues and the luminous light of the divine",
        "artwork where light itself becomes the subject radiating from a sacred center outward",
        
        # Referencias estilísticas poderosas
        "El Greco painting of elongated mystical figures in supernatural light reaching toward heaven",
        "Baroque painting of divine light breaking through clouds onto kneeling worshipping figures",
        
        # Espiritualidad universal no cristiana
        "painting of a Buddhist monk in deep meditation dissolving into universal consciousness",
        "artwork of a Sufi dancer spinning in sacred trance merging with the divine through movement",
    ],
    "gratitud": [
        # Ancla fuerte - gratitud requiere recepción emocional visible, no trabajo
        "painting of a figure weeping with relief and joy at having received unexpected help from another",
        "artwork of someone looking upward with open emotional expression after receiving grace not laboring",
        
        # El gesto físico universal de gratitud - blindado contra labor
        "painting of a figure bowing deeply with hands pressed together in reverent thankfulness toward another",
        "artwork of a person dropping to their knees overwhelmed by gratitude for something just received",
        "painting of hands clasped in prayer of thanksgiving raised toward a source of given grace",
        
        # Momentos concretos de recibir - siempre hay un dador visible
        "artwork of a starving figure receiving bread from a stranger with tears of profound relief",
        "painting of a sick person healed reaching gratefully toward the hands that helped them",
        "artwork of a refugee welcomed with open arms after a long hard journey by someone waiting",
        
        # Gratitud hacia la naturaleza - blindada contra flores como carga laboral
        "painting of a farmer kneeling in an empty harvested field giving thanks with arms raised upward",
        "artwork of a figure with arms open facing sunrise in silent morning gratitude for being alive",
        
        # Gratitud intergeneracional
        "artwork of a child embracing a grandparent with genuine heartfelt appreciation and love",
        "painting of a student honoring a beloved teacher who changed the course of their entire life",
        
        # Paleta y composición
        "painting where one figure is lower bowed toward another specific person in natural deference",
        "artwork with warm golden light falling on the emotional moment of giving and receiving between two",
        
        # Referencias estilísticas
        "Rembrandt painting of the Prodigal Son received with overwhelming compassionate gratitude",
        "Jean-François Millet painting of peasants pausing work in humble prayer of thanksgiving",
        
        # Gratitud silenciosa
        "painting of a solitary figure alone at dusk quietly counting the gifts of an ordinary day",
        "artwork of a person holding something simple and precious with complete appreciative stillness",
    ],
   "fe": [
        # El acto de confiar sin ver - único de fe
        "painting of a figure walking calmly forward through complete darkness trusting the unseen path",
        "artwork of a person stepping off a cliff edge in complete trust with serene unafraid expression",
        "painting of a figure releasing something precious upward into darkness trusting it will be held",
        
        # Fe en la adversidad extrema
        "artwork of a lone figure kneeling in prayer amid ruins disaster or overwhelming devastation",
        "painting of a person maintaining complete inner calm while everything around them collapses",
        "artwork of a martyr or saint facing death with absolute unshakeable peaceful certainty",
        
        # Gestos físicos específicos de fe
        "painting of hands releasing a dove upward in complete trust it will find its way home",
        "artwork of a figure with eyes closed arms open completely surrendered in total trust",
        "painting of a person crossing a rickety bridge in darkness holding only a small candle",
        
        # Fe colectiva y comunidad
        "artwork of a congregation in humble unified prayer with bowed heads and folded hands",
        "painting of pilgrims on a long difficult sacred journey sustained only by shared belief",
        "artwork of people kneeling together in a simple chapel with profound collective devotion",
        
        # Paleta específicamente de fe
        "painting moving from deep darkness toward a single distant point of unwavering light ahead",
        "artwork where a tiny flame or light source holds steady against overwhelming surrounding darkness",
        
        # Referencias estilísticas poderosas
        "Caravaggio painting of a figure receiving divine calling with overwhelming surrender and trust",
        "George Frederic Watts allegorical painting of blind faith clinging to hope in total darkness",
        
        # Fe como acto cotidiano silencioso
        "painting of an ordinary person in simple daily prayer with complete undemonstrative quiet trust",
        "artwork of a weathered face in prayer that has prayed the same prayer ten thousand times before",
    ],
   "resiliencia": [
        # El momento de levantarse - único de resiliencia
        "painting of a battered figure slowly rising from the ground after being knocked completely down",
        "artwork of a person covered in wounds and dust choosing to stand up one more time",
        "painting of a fighter or survivor on one knee gathering strength to rise again refusing defeat",
        
        # Reconstrucción después del daño
        "artwork of hands carefully rebuilding something that was completely destroyed piece by piece",
        "painting of a community rebuilding their homes after fire flood or devastating destruction",
        "artwork of a garden growing back stronger through burned scorched earth after a wildfire",
        
        # Cicatrices como evidencia de resiliencia
        "painting of a figure whose scars and wounds are worn with quiet pride not hidden with shame",
        "artwork of a weathered face that has survived much and carries that survival with dignity",
        "painting of a tree split by lightning still alive still growing defiantly around the damage",
        
        # Naturaleza como metáfora de resiliencia
        "artwork of a lone flower blooming through cracked concrete after a long devastating winter",
        "painting of a forest regenerating with new green growth emerging from the ash of destruction",
        "artwork of a river finding a new path around an obstacle that completely blocked its course",
        
        # Paleta específica de resiliencia
        "painting transitioning from dark damaged lower tones to new growth and light in upper composition",
        "artwork where signs of damage and new life coexist in the same figure landscape or object",
        
        # Referencias estilísticas poderosas
        "Käthe Kollwitz painting of working class figures enduring loss grief and rising with quiet strength",
        "painting in the tradition of Goya showing human survival and dignity after unimaginable suffering",
        
        # Resiliencia silenciosa y cotidiana
        "painting of an ordinary person continuing their daily life with quiet determination after great loss",
        "artwork of a figure who has lost much but tends carefully to what little still remains with love",
    ],
    "ira": [
        # El rostro humano como expresión máxima de ira
        "painting of a face contorted in pure rage with veins bulging jaw clenched and eyes blazing",
        "artwork of a figure screaming in anguished fury with every muscle in their face and neck tense",
        "painting of eyes burning with cold controlled fury more terrifying than open explosive rage",
        
        # El gesto físico de ira explosiva
        "artwork of a figure smashing and destroying furniture in blind uncontrolled destructive fury",
        "painting of an enraged person overturning a table with fists clenched and teeth bared in anger",
        "artwork of two figures in a violent angry confrontation with hateful expressions and aggressive postures",
        
        # Ira justa ante la injusticia
        "painting of a wronged figure confronting their oppressor with righteous unstoppable fury",
        "artwork of a parent in fierce protective rage defending their child from imminent threat",
        "painting of a crowd erupting in collective fury at a visible act of profound injustice",
        
        # Ira contenida más peligrosa que la explosiva
        "artwork of a figure sitting perfectly still with rage barely contained behind calm exterior",
        "painting of clenched hands hidden under a table while a face maintains dangerous composure",
        "artwork of a figure turned away from the viewer shoulders rigid with suppressed volcanic fury",
        
        # Naturaleza como espejo de ira
        "dark painting of a violent storm destroying houses and uprooting trees in wrathful devastation",
        "artwork of a blood red sky over a battlefield with smoke and ruins expressing collective human rage",
        
        # Referencias estilísticas poderosas
        "Goya dark painting of savage violence showing the terrifying face of human rage and brutality",
        "Expressionist painting with dark anguished distorted faces expressing suffering and emotional fury",
        
        # Ira y consecuencias
        "painting of the devastating aftermath of rage with broken objects scattered and silence returned",
        "artwork of a figure alone after an explosion of anger surrounded by the damage they have caused",
    ],
   "miedo": [
        # La respuesta corporal específica al miedo
        "painting of a figure frozen in terror with eyes wide open mouth agape in silent scream",
        "artwork of a person pressing themselves against a wall trying to disappear into the shadows",
        "painting of a figure running desperately away from something unseen but terrifyingly close",
        
        # El miedo en el rostro humano
        "artwork of a face pale with terror eyes reflecting something horrifying just out of frame",
        "painting of a child hiding under blankets with only terrified eyes visible in the darkness",
        "artwork of a figure looking over their shoulder at something that may or may not be there",
        
        # Amenaza inminente y concreta
        "painting of a small figure pursued through a dark forest by an unseen threatening presence",
        "artwork of a person trapped in a corner with no escape route and danger closing in fast",
        "painting of a crowd fleeing in blind panic from an overwhelming approaching catastrophe",
        
        # Miedo psicológico e interior
        "artwork of a figure alone in darkness surrounded by shadows that may or may not be real",
        "painting of a person waking from a nightmare still unsure if the terror has truly ended",
        "artwork of a solitary figure at a window seeing their own reflection merged with something wrong",
        
        # Naturaleza amenazante
        "painting of a lone figure in a vast dark forest at night with eyes glowing between the trees",
        "artwork of a stormy sea with a tiny vessel about to be swallowed by an enormous dark wave",
        
        # Referencias estilísticas poderosas
        "Munch style painting of a figure consumed by existential terror in a distorted anxious landscape",
        "Fuseli painting of a nightmare figure crouching on a sleeping person's chest in the darkness",
        
        # Miedo colectivo
        "painting of a crowd of faces all turned in the same direction frozen in collective shared terror",
        "artwork of people sheltering together trembling as something terrible passes just outside",
        
        # El momento antes
        "painting capturing the unbearable silence just before something terrifying is about to happen",
        "artwork of a hand reaching for a door handle knowing something wrong waits on the other side",
    ],
   "humildad": [
        # Ancla nueva - humildad de quien no tiene opción, pobreza con dignidad
        "painting of a poor laborer kneeling under an enormous load accepting their humble station with dignity",
        "artwork of an indigenous or working class figure bent under the weight of poverty with quiet acceptance",
        "painting of a barefoot worker crouching on the ground small and invisible doing necessary work alone",
        "Diego Rivera style painting of a humble Mexican worker carrying goods with dignified silent acceptance",
        
        # El gesto de achicarse voluntariamente
        "artwork of a king or leader bowing their head before an ordinary person with genuine respect",
        "painting of a wise elder sitting at the feet of someone younger listening with complete attention",
        
        # Simplicidad elegida sobre ostentación
        "artwork of a person of means choosing to eat a simple meal alone in an unadorned modest space",
        "painting of a scholar or leader dressed in plain simple clothes amid symbols of available wealth",
        
        # Trabajo humilde realizado con dignidad - blindado contra ternura
        "painting of a solitary person scrubbing floors sweeping streets doing invisible work with no audience",
        "artwork of rough calloused hands doing simple humble labor with unhurried dignified attention",
        "painting of Jean-François Millet peasants bowing heads in humble prayer over simple daily bread",
        
        # Humildad ante la naturaleza
        "artwork of a tiny human figure completely dwarfed by the immensity of stars mountains or ocean",
        "painting of a person sitting quietly in nature aware of their smallness with peaceful acceptance",
        
        # Paleta y composición
        "painting where the main figure occupies the lowest smallest position in the composition by necessity",
        "artwork in muted earth tones, plain textures and the quiet beauty of unadorned simple spaces",
        
        # Referencias estilísticas
        "Rembrandt painting of Christ washing disciples feet with tender unhurried humble service",
        "Jean-François Millet painting of peasant laborers in simple dignified humble daily existence",
        
        # Humildad como escucha
        "painting of a figure listening with complete absorbed attention to someone society would ignore",
        "artwork of a person stepping aside to let another take the recognition they themselves deserve",
    ],
    "conocimiento": [
        # Core concept - directo y fuerte
        "painting symbolizing knowledge, wisdom and intellectual enlightenment",
        
        # Elementos visuales clásicos asociados al conocimiento
        "artwork depicting books, scrolls, manuscripts and the pursuit of learning",
        "painting of scholars, philosophers or thinkers in contemplation",
        "art showing light illuminating darkness as a metaphor for knowledge",
        
        # Símbolos universales
        "painting with symbols of wisdom: owls, candles, globes, astronomical instruments",
        "artwork depicting libraries, study rooms or academic settings",
        
        # Filosófico / abstracto
        "allegorical painting representing truth, reason and human understanding",
        "art celebrating the power of the human mind and intellectual discovery",
        
        # Narrativo / histórico
        "painting of ancient philosophers, scientists or scholars sharing knowledge",
        "Renaissance or classical painting depicting education and intellectual virtue",
        
        # Emocional / contemplativo
        "meditative artwork exploring the depth of thought and inner wisdom",
        "painting capturing the moment of insight, discovery or enlightenment",
    ],
    "belleza": [
        # Belleza del cuerpo humano idealizado
        "painting of a perfectly proportioned classical nude figure in harmonious graceful repose",
        "artwork of a face of extraordinary classical beauty rendered with meticulous idealized perfection",
        "painting of dancers or figures in movement where the human form achieves pure aesthetic grace",
        
        # Belleza de la naturaleza en su momento cumbre
        "artwork of a landscape at the precise moment of perfect golden light just before sunset",
        "painting of cherry blossoms falling over still water in a moment of ephemeral perfect beauty",
        "artwork of a single perfect flower rendered with botanical precision and aesthetic reverence",
        
        # Belleza arquitectónica y artificial
        "painting of classical Greek or Renaissance architecture in perfect harmonious proportion",
        "artwork of an ornate interior with perfect symmetry gilded surfaces and harmonious design",
        "painting of a still life arranged with such perfection it transcends mere objects into pure beauty",
        
        # Belleza como experiencia estética consciente
        "artwork of a figure standing before a painting or sculpture overwhelmed by its beauty",
        "painting of a person pausing before a sunset unable to move consumed by pure aesthetic wonder",
        "artwork of an artist stepping back from their work seeing for the first time it has become beautiful",
        
        # Paleta y composición que CLIP asocia a belleza pura
        "classical painting with soft delicate colors and gentle harmonious composition evoking pure beauty",
        "artwork of serene natural beauty with soft light gentle curves and ethereal peaceful atmosphere",
        
        # Referencias estilísticas poderosas
        "Botticelli painting of ideal feminine beauty emerging from nature with divine harmonious grace",
        "Vermeer painting where ordinary domestic scene achieves transcendent beauty through perfect light",
        
        # Belleza efímera y melancólica
        "painting of something beautiful in the exact moment it begins to fade wither or disappear",
        "artwork capturing the bittersweet beauty of a perfect moment that cannot last and will not return",
    ]
}

# Keywords to detect emotional queries (Spanish)
_EMOTION_VERB_KEYWORDS: Set[str] = {
    "inspire", "inspira", "inspirar", "inspiración",
    "evoque", "evocar", "evoca", "evocación",
    "transmita", "transmitir", "transmite",
    "sensación", "sentimiento", "emoción", "emociones",
    "que exprese", "expresar", "expresa",
    "ambiente", "atmósfera", "mood",
}

# Automatically include all emotion category names as keywords
EMOTION_KEYWORDS_ES: Set[str] = _EMOTION_VERB_KEYWORDS | {
    name.lower() for name in EMOTION_CATEGORIES.keys()
}
