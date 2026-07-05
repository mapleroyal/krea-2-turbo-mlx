from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "krea-2-turbo-mlx"
PYTHON_PACKAGE = "krea_2_turbo_mlx"
ARTIFACT_FORMAT = "krea-2-turbo-mlx-artifact"

OFFICIAL_HF_REPO_ID = "krea/Krea-2-Turbo"
OFFICIAL_HF_REVISION = "1161245028ef398cd0a951101b2bbf486464f841"
OFFICIAL_HF_URL = f"https://huggingface.co/{OFFICIAL_HF_REPO_ID}"
OFFICIAL_LICENSE_URL = f"{OFFICIAL_HF_URL}/blob/main/LICENSE.pdf"
OFFICIAL_KREA_INFERENCE_REPO = "https://github.com/krea-ai/krea-2"

EXPECTED_SOURCE_COMPONENTS = (
    "model_index",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer",
    "vae",
)
DIFFUSERS_COMPONENTS = (
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer",
    "vae",
)
EXPECTED_COMPONENT_CLASSES = {
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3VLModel"],
    "tokenizer": ["transformers", "Qwen2Tokenizer"],
    "transformer": ["diffusers", "Krea2Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLQwenImage"],
}
EXPECTED_SOURCE_METADATA_PATHS = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "vae/config.json",
)

DEFAULT_SOURCE_DIR = Path("models") / "Krea-2-Turbo"
DEFAULT_ARTIFACT_PATH = Path("artifacts") / "krea-2-turbo-mlx"
DEFAULT_CONFIG_PATH = Path(".krea-2-turbo-mlx") / "config.json"
DEFAULT_GUI_PORT = 8765
DEFAULT_GUI_SETTINGS_FILENAME = "gui-settings.json"
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_LORA_DIR = Path("loras")
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "krea-2-turbo.png"
DEFAULT_OUTPUT_TEMPLATE = (
    "krea-{batch_index:04d}-p{prompt_index:02d}-i{image_index:02d}-seed{seed}.png"
)
DEFAULT_GUI_LAUNCHER = "Launch Krea 2 Turbo.command"

TEXT_ENCODER_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
KREA_TEXT_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
KREA_TEXT_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
KREA_TEXT_PREFIX_DROP_INDEX = 34
KREA_TEXT_SUFFIX_TOKEN_COUNT = 5
DEFAULT_TEXT_MAX_SEQUENCE_LENGTH = 512
TEXT_ENCODER_WEIGHT_BODY_PREFIX = "language_model."
DEFAULT_GENERATION_STEPS = 8
DEFAULT_GUIDANCE_SCALE = 0.0
DEFAULT_DISTILLED_SHIFT = 1.15
DEFAULT_GENERATION_WIDTH = 1024
DEFAULT_GENERATION_HEIGHT = 1024
OUTPUT_ALIGNMENT = 16
MAX_GENERATION_SIZE = 2048
MAX_GENERATION_SEED = (2**32) - 1

FULL_PRECISION_ONLY = True
OFFICIAL_SOURCE_DTYPES = {
    "transformer": ("BF16", "F32"),
    "text_encoder": ("BF16",),
    "vae": ("F32",),
}
