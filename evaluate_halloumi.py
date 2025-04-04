# evaluate_halloumi.py

import os
import argparse
import logging
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import accelerate
from sklearn.metrics import f1_score, balanced_accuracy_score

# --- Configuration ---
EVAL_DATASET = "oumi-ai/oumi-groundedness-benchmark"
EVAL_SPLIT = "test"
PROMPT_COLUMN = "halloumi 8b prompt"
LABEL_COLUMN = "label"
MAX_MODEL_LENGTH = 8192
GENERATION_MAX_NEW_TOKENS = 512
TORCH_DTYPE_STR = "bfloat16"
TOKENIZER_PAD_TOKEN = "<|finetune_right_pad_id|>"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def load_model_and_tokenizer(model_path: str, torch_dtype_str: str):
    """Loads the fine-tuned model and tokenizer directly from the specified path."""
    logger.info(f"Loading fine-tuned model from: {model_path}")

    # Determine torch dtype
    if torch_dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        logger.info("Using bfloat16.")
    elif torch_dtype_str == "float16":
        torch_dtype = torch.float16
        logger.info("Using float16.")
    else:
        torch_dtype = torch.float32
        logger.info("Using float32.")

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="sdpa",
    )
    logger.info("Fine-tuned model loaded.")

    # Load the tokenizer
    logger.info(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left'
    )

    # Handle custom pad token
    if tokenizer.pad_token is None or tokenizer.pad_token != TOKENIZER_PAD_TOKEN:
        logger.warning(f"Saved tokenizer pad token is '{tokenizer.pad_token}', expected '{TOKENIZER_PAD_TOKEN}'.")
        logger.info(f"Attempting to set pad token to: {TOKENIZER_PAD_TOKEN}")
        if TOKENIZER_PAD_TOKEN not in tokenizer.get_vocab():
            logger.info(f"Adding new pad token {TOKENIZER_PAD_TOKEN} to tokenizer vocab.")
            tokenizer.add_special_tokens({'pad_token': TOKENIZER_PAD_TOKEN})
            model.resize_token_embeddings(len(tokenizer))
            logger.info("Resized model token embeddings.")
        tokenizer.pad_token = TOKENIZER_PAD_TOKEN
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        logger.info(f"Tokenizer pad token '{TOKENIZER_PAD_TOKEN}' loaded correctly.")

    # Set model to evaluation mode
    model.eval()

    return model, tokenizer

def load_evaluation_data(dataset_name: str, split: str, num_examples: int | None, shuffle_seed: int | None = 42):
    """Loads the dataset, optionally shuffles, and prepares prompts and labels."""
    logger.info(f"Loading dataset: {dataset_name}, split: {split}")
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}' split '{split}'. Error: {e}")
        logger.error("Please ensure the dataset exists and you have internet access.")
        return None, None

    # Add Shuffling Step
    if shuffle_seed is not None:
        logger.info(f"Shuffling dataset with seed: {shuffle_seed}")
        try:
            dataset = dataset.shuffle(seed=shuffle_seed)
            logger.info("Dataset shuffled successfully.")
        except Exception as e:
            logger.error(f"Failed to shuffle dataset: {e}", exc_info=True)
            logger.warning("Proceeding with unshuffled dataset due to shuffling error.")
    else:
        logger.info("Dataset not shuffled (no seed provided).")

    # Limit examples if requested
    if num_examples is not None and num_examples > 0:
        if num_examples < len(dataset):
            dataset = dataset.select(range(num_examples))
            logger.info(f"Selected the first {num_examples} examples from the (potentially shuffled) dataset.")
        else:
            logger.warning(f"Requested {num_examples} examples, but dataset only has {len(dataset)}. Using all.")

    # Extract prompts and labels
    try:
        prompts = dataset[PROMPT_COLUMN]
        labels_str = dataset[LABEL_COLUMN]
    except KeyError as e:
        logger.error(f"Failed to find column '{e}' in the dataset. Available columns: {dataset.column_names}")
        return None, None

    # Convert string labels to numeric (0 for SUPPORTED, 1 for UNSUPPORTED)
    labels_numeric = [0 if label.upper() == "SUPPORTED" else 1 for label in labels_str]
    logger.info(f"Loaded {len(prompts)} prompts and corresponding labels.")

    return prompts, labels_numeric

def _extract_prediction(response: str) -> int:
    """
    Extracts hallucination prediction, explicitly checking both tags.
    Returns: 0 if supported, 1 if unsupported. Logs warnings.
    """
    response_lower = response.lower()
    has_unsupported = "<|unsupported|>" in response_lower or "<unsupported>" in response_lower
    has_supported = "<|supported|>" in response_lower or "<supported>" in response_lower

    return 1 if has_unsupported else 0

def run_evaluation(model, tokenizer, prompts: list[str], labels: list[int], batch_size: int):
    """Runs inference and calculates metrics."""
    predictions = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    logger.info(f"Starting inference on {len(prompts)} examples in {num_batches} batches (batch size: {batch_size}).")

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating Batches"):
            batch_prompts = prompts[i:i+batch_size]

            # Tokenize the batch
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_MODEL_LENGTH - GENERATION_MAX_NEW_TOKENS,
                return_attention_mask=True,
            ).to(model.device)

            # Generate responses
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            except Exception as e:
                logger.error(f"Error during model.generate in batch starting at index {i}: {e}")
                predictions.extend([-1] * len(batch_prompts))
                continue

            # Decode generated sequences
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Extract predictions from decoded outputs
            for decoded_idx, decoded in enumerate(decoded_outputs):
                original_prompt_index = i + decoded_idx
                original_prompt = prompts[original_prompt_index]

                # Extract the generated part of the response
                input_ids_len = inputs.input_ids[decoded_idx].shape[0]
                generated_ids = outputs[decoded_idx][input_ids_len:]
                generated_part = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                # Fallback to find generated part if primary method fails
                if not generated_part:
                    logger.warning(f"Primary generation extraction failed for prompt index {original_prompt_index}. Trying fallback method.")
                    try:
                        prompt_in_decoded_index = decoded.rindex(original_prompt)
                        start_generation = prompt_in_decoded_index + len(original_prompt)
                        generated_part = decoded[start_generation:].strip()
                    except ValueError:
                        logger.warning(f"Fallback failed. Could not reliably isolate generated text from prompt in response: '{decoded[:100]}...'. Using full decoded text for prediction.")
                        generated_part = decoded

                if not generated_part:
                    logger.warning(f"Empty generated part detected for prompt index {original_prompt_index}. Full decoded: '{decoded[:100]}...'")
                    predictions.append(-1)
                    continue

                prediction = _extract_prediction(generated_part)
                predictions.append(prediction)

    # Filter out error predictions (-1) before calculating metrics
    valid_indices = [idx for idx, p in enumerate(predictions) if p != -1]
    if not valid_indices:
        logger.error("No valid predictions were generated. Cannot calculate metrics.")
        return None

    filtered_labels = [labels[i] for i in valid_indices]
    filtered_predictions = [predictions[i] for i in valid_indices]

    logger.info(f"Generated {len(filtered_predictions)} valid predictions out of {len(labels)} total examples.")

    # Calculate metrics
    if not filtered_labels:
        logger.error("No valid labels remain after filtering predictions. Cannot calculate metrics.")
        return None

    try:
        f1 = f1_score(filtered_labels, filtered_predictions, average="macro", zero_division=0)
        bacc = balanced_accuracy_score(filtered_labels, filtered_predictions)
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return None

    return {"f1_macro": f1, "balanced_accuracy": bacc}

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Llama model (Full Model Load).")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the directory containing the full fine-tuned model files.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of examples from the test set to evaluate on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference.",
    )
    args = parser.parse_args()

    # Validate model path
    if not os.path.isdir(args.model_path):
        logger.error(f"Model path not found or is not a directory: {args.model_path}")
        return
    expected_files = ['config.json', 'model.safetensors', 'tokenizer.json']
    if not all(os.path.isfile(os.path.join(args.model_path, f)) for f in expected_files):
        logger.warning(f"Model directory '{args.model_path}' might be missing expected files.")

    # 1. Load Model and Tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path, TORCH_DTYPE_STR)
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer from {args.model_path}: {e}", exc_info=True)
        return

    # 2. Load Data
    prompts, labels = load_evaluation_data(EVAL_DATASET, EVAL_SPLIT, args.num_examples)
    if prompts is None or labels is None:
        logger.error("Failed to load or process evaluation data. Exiting.")
        return
    if not prompts:
        logger.error("No data loaded for evaluation. Check dataset name, split, and num_examples.")
        return

    # 3. Run Evaluation
    try:
        results = run_evaluation(model, tokenizer, prompts, labels, args.batch_size)
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA Out of Memory during evaluation!")
        logger.error(f"Try reducing --batch_size (current: {args.batch_size}).")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during evaluation: {e}", exc_info=True)
        return

    # 4. Print Results
    if results:
        logger.info("--- Evaluation Results ---")
        logger.info(f"  Model Path: {args.model_path}")
        logger.info(f"  Dataset: {EVAL_DATASET} (split: {EVAL_SPLIT})")
        logger.info(f"  Examples Evaluated: {len(prompts)}")
        logger.info(f"  Macro F1 Score:          {results['f1_macro']:.4f}")
        logger.info(f"  Balanced Accuracy:       {results['balanced_accuracy']:.4f}")
        logger.info("-------------------------")
    else:
        logger.error("Evaluation finished without producing results.")

if __name__ == "__main__":
    main()