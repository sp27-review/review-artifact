"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


DATA_DIR = ROOT / "data"
DISC_DIR = DATA_DIR / "DISC"
DISC_JSON = DISC_DIR / "DISC.json"


MODEL_DIR = ROOT / "Model"
PURPLE_LLAMA_DIR = MODEL_DIR / "PurpleLlama"
CLASSIFYIT_DIR = PURPLE_LLAMA_DIR / "SensitiveDocClassification"
OLLAMA_VERSIONS_DIR = MODEL_DIR / "ollama_versions"
OLLAMA_BIN = OLLAMA_VERSIONS_DIR / "v0.24.0" / "bin" / "ollama"

CHECKPOINTS_DIR = ROOT / "checkpoints"
OLLAMA_MODELS_DIR = CHECKPOINTS_DIR / "ollama" / "models"


EXPERIMENTS_DIR = ROOT / "_experiments"
SHARED_EXPERIMENT_DIR = EXPERIMENTS_DIR / "shared"
SHARED_CONFIGS_DIR = SHARED_EXPERIMENT_DIR / "configs"
LLAMA_STACK_CONFIG = SHARED_CONFIGS_DIR / "llama_stack_ollama.yaml"

DISC_EXPERIMENT_DIR = EXPERIMENTS_DIR / "main1_DISC"
MAIN_EXPERIMENT_DIR = DISC_EXPERIMENT_DIR
MAIN_RESULTS_DIR = MAIN_EXPERIMENT_DIR / "results"
MAIN_RESULTS_LOGS_DIR = MAIN_RESULTS_DIR / "logs"
ABLATION_01_DIR = DISC_EXPERIMENT_DIR / "ablation_01"
OCRTEXT_CHUNK_EXPLORATORY_DIR = MAIN_RESULTS_DIR / "ocrtext_chunk_exploratory"


IA_CIA_EXPERIMENT_DIR = EXPERIMENTS_DIR / "main2_IA_CIA"
IA_CIA_DIR = IA_CIA_EXPERIMENT_DIR / "data" / "ia_cia_readingroom"
IA_CIA_RAW_OCR_DIR = IA_CIA_DIR / "raw_ocr"
IA_CIA_MANIFESTS_DIR = IA_CIA_DIR / "manifests"
IA_CIA_DERIVED_DIR = IA_CIA_DIR / "derived"


EVIDENCE_EXPERIMENT_DIR = EXPERIMENTS_DIR / "main3_evidence"
EVIDENCE_RESULTS_DIR = EVIDENCE_EXPERIMENT_DIR / "results"


RESULT_BOX_DIR = ROOT / "result_box"
RUN_LOGS_DIR = RESULT_BOX_DIR / "run_logs"
