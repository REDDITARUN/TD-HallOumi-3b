# TD-HallOumi-3B: Training and Evaluation Code

This repository contains the code used to fine-tune and evaluate the `TEEN-D/TD-HallOumi-3B` model, a Llama-3.2-3B-Instruct model specialized for **Claim Verification / Hallucination Detection**.

The goal of this project was to develop a reliable open-source model capable of assessing whether claims made in a text are supported by a given context document, leveraging the datasets and methodologies pioneered by the [Oumi AI HallOumi project](https://oumi.ai/blog/posts/introducing-halloumi).

**➡️ Find the Fine-tuned Model on Hugging Face:** [TEEN-D/TD-HallOumi-3B](https://huggingface.co/TEEN-D/TD-HallOumi-3B)

## Performance

Evaluated on the [oumi-ai/oumi-groundedness-benchmark](https://huggingface.co/datasets/oumi-ai/oumi-groundedness-benchmark) for Hallucination Detection (Macro F1 Score):


![image](https://github.com/user-attachments/assets/59cad708-88a7-4422-b7f7-233e9886c460)


*   **TD-HallOumi-3B\*** achieves **68.00%** Macro F1.
*   **Highly Efficient:** This 3B parameter model outperforms larger models like Open AI o1, Llama 3.1 405B and Gemini 1.5 Pro.
*   **Competitive:** Ranks closely behind Claude Sonnet 3.5 (69.60%).

This model offers strong hallucination detection capabilities with significantly fewer parameters than many alternatives.

## Repository Contents

This repository includes:

*   **`train.yaml`**: The configuration file used with the Oumi framework (`oumi train`) to perform Supervised Fine-Tuning (SFT) using LoRA on the `meta-llama/Llama-3.2-3B-Instruct` model.
*   **`evaluate_halloumi.py`**: A Python script to evaluate the fine-tuned model's performance on the claim verification task using the `oumi-ai/oumi-groundedness-benchmark` dataset.

## What We Tried to Develop

We aimed to create an accessible and effective 3B parameter model for identifying hallucinations (unsupported claims) in text relative to provided source documents. This involved:
1.  Fine-tuning the powerful `meta-llama/Llama-3.2-3B-Instruct` model.
2.  Utilizing a curated mix of Oumi AI datasets specifically designed for claim verification.
3.  Employing efficient fine-tuning techniques (LoRA via the Oumi framework).
4.  Providing evaluation code to benchmark the model's performance on a relevant task-specific dataset.

## Prerequisites

Before running the code in this repository, ensure you have the following installed:

1.  **Python:** Version 3.9 or later.
2.  **Git:** Required for cloning repositories.
3.  **pip:** Python package installer.
4.  **Virtual Environment (Recommended):** To avoid dependency conflicts.
    ```bash
    # Linux / MacOS
    python -m venv .env
    source .env/bin/activate

    # Windows (use Git Bash or WSL)
    # python -m venv .env
    # .env\Scripts\activate
    ```
5.  **Oumi Framework:** The training process relies on the Oumi framework. Install it within your virtual environment:
    ```bash
    pip install oumi
    # Or, for GPU support (Nvidia/AMD):
    # pip install oumi[gpu]
    ```
    *(Note: Oumi may not install on Intel Macs due to PyTorch limitations).*
6.  **Core Python Libraries:** Install necessary libraries for evaluation (and implicitly for Oumi/training):
    ```bash
    pip install torch transformers datasets accelerate scikit-learn tqdm
    # Consider creating a requirements.txt file for easier installation
    ```
7.  **Git LFS:** If you plan to download large model files locally, ensure Git LFS is installed:
    *   Install from [https://git-lfs.github.com/](https://git-lfs.github.com/).
    *   Run `git lfs install` once after installation.

## How to Use

### 1. Training (Replicating the Fine-tuning)

The fine-tuning process uses the Oumi framework's training command with the provided `train.yaml` configuration file.

1.  **Set up** your environment as described in Prerequisites.
2.  **Review `train.yaml`**:
    *   Ensure the `model.model_name` points to the correct base model (`meta-llama/Llama-3.2-3B-Instruct`). You might need to accept terms on its Hugging Face page first.
    *   Verify the `data.train.datasets` and `data.validation.datasets` sections list the correct Oumi datasets.
    *   **Important:** Modify the `training.output_dir` parameter to specify where the trained model artifacts (including the final merged model specified by `save_final_model: True`) should be saved on your system.
3.  **Run Training:** Execute the following command from the root of this repository:
    ```bash
    oumi train -c train.yaml
    ```
    This will initiate the SFT process using the parameters defined in the YAML file (LoRA configuration, datasets, learning rate, batch size, etc.). Training requires suitable hardware (GPU recommended). The final merged model weights will be saved in the specified `output_dir`.

### 2. Evaluation

The `evaluate_halloumi.py` script evaluates the fine-tuned model on the `oumi-ai/oumi-groundedness-benchmark` test set.

1.  **Ensure** you have the fine-tuned model available locally. This can be:
    *   The model saved in the `output_dir` after running the training step above.
    *   The model downloaded from the Hugging Face Hub: `huggingface-cli download TEEN-D/TD-HallOumi-3B --local-dir ./TD-HallOumi-3B` (or use Python's `snapshot_download`).
2.  **Run Evaluation Script:** Execute the script from the command line:
    ```bash
    python evaluate_halloumi.py --model_path <path_to_your_model_directory> [--batch_size <bs>] [--num_examples <n>]
    ```
    *   `--model_path <path_to_your_model_directory>`: **(Required)** Path to the local directory containing the *full* fine-tuned model files (e.g., `./TD-HallOumi-3B` if downloaded, or the `output_dir` from training).
    *   `--batch_size <bs>`: (Optional) Batch size for inference during evaluation. Adjust based on your GPU memory. Defaults to `4`.
    *   `--num_examples <n>`: (Optional) Evaluate only the first `n` examples from the benchmark's test set (after shuffling). If omitted, the script will evaluate the **entire** `oumi-ai/oumi-groundedness-benchmark` test split.

    The script will load the model and tokenizer, process the evaluation dataset, generate predictions (`<|supported|>` or `<|unsupported|>`), and print the Macro F1 Score and Balanced Accuracy.

## License

The code in this repository (e.g., `evaluate_halloumi.py`) is released under the [Apache 2.0 License].

Note that the underlying components have their own licenses:
*   **Base Model (`meta-llama/Llama-3.2-3B-Instruct`):** Subject to the Llama 3.2 Community License.
*   **Oumi Framework:** Apache 2.0 License.
*   **Datasets:** Licenses vary (Llama 3.1 Community License, CC-BY-NC-4.0) as specified in the Citations section and on their respective Hugging Face dataset cards. Please ensure compliance with all applicable licenses.

## Acknowledgements

*   This work heavily relies on the **Llama 3.2** model series by Meta AI.
*   We thank the **Oumi AI** team for their pioneering work on the HallOumi project, the Oumi framework, and for curating and open-sourcing the essential datasets used for training and evaluation.

## Citations

If you use this model, please consider citing the base model, the datasets, the Oumi AI HallOumi project, and this specific fine-tuned model artifact:

**This Fine-tuned Model & Code:**
```bibtex
@misc{teen_d_halloumi_3b_2024,
  author = {{Tarun Reddi}}, # Or Your Name/Team Name
  title = {TD-HallOumi-3B: Training and Evaluation Code for Llama-3.2-3B-Instruct Claim Verification},
  year = {2024}, # Adjust year if needed
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{<URL_of_this_GitHub_Repo>}} # <-- ADD GITHUB REPO URL HERE
}

@misc{teen_d_halloumi_3b_model_2024,
  author = {{Tarun Reddi}}, # Or Your Name/Team Name
  title = {TD-HallOumi-3B: Fine-tuned Llama-3.2-3B-Instruct for Claim Verification},
  year = {2024}, # Adjust year if needed
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\url{https://huggingface.co/TEEN-D/TD-HallOumi-3B}}
}

Base Model:

@misc{meta2024llama32,
  title = {Introducing Llama 3.2: The Next Generation of Open Weights AI Models},
  author = {Meta AI},
  year = {2024},
  url = {https://ai.meta.com/blog/llama-3-2-ai-models/}
}

Datasets:

@misc{oumiANLISubset,
  author = {Jeremiah Greer},
  title = {Oumi ANLI Subset},
  month = {March},
  year = {2025},
  url = {https://huggingface.co/datasets/oumi-ai/oumi-anli-subset}
}

@misc{oumiC2DAndD2CSubset,
  author = {Jeremiah Greer},
  title = {Oumi C2D and D2C Subset},
  month = {March},
  year = {2025},
  url = {https://huggingface.co/datasets/oumi-ai/oumi-c2d-d2c-subset}
}

@misc{oumiSyntheticClaims,
  author = {Jeremiah Greer},
  title = {Oumi Synthetic Claims},
  month = {March},
  year = {2025},
  url = {https://huggingface.co/datasets/oumi-ai/oumi-synthetic-claims}
}

@misc{oumiSyntheticDocumentClaims,
  author = {Jeremiah Greer},
  title = {Oumi Synthetic Document Claims},
  month = {March},
  year = {2025},
  url = {https://huggingface.co/datasets/oumi-ai/oumi-synthetic-document-claims}
}

@misc{oumiGroundednessBenchmark,
  author = {Jeremiah Greer},
  title = {Oumi Groundedness Benchmark},
  month = {March},
  year = {2025},
  url = {https://huggingface.co/datasets/oumi-ai/oumi-groundedness-benchmark}
}


Oumi Platform & HallOumi Project:

@software{oumi2025,
  author = {Oumi Community},
  title = {Oumi: an Open, End-to-end Platform for Building Large Foundation Models},
  month = {January},
  year = {2025},
  url = {https://github.com/oumi-ai/oumi}
}

@article{halloumi2025,
  author = {Greer, Jeremiah and Koukoumidis, Manos and Aisopos, Konstantinos and Schuler, Michael},
  title = {Introducing HallOumi: A State-of-the-Art Claim-Verification Model},
  journal = {Oumi AI Blog},
  year = {2025},
  month = {April},
  url = {https://oumi.ai/blog/posts/introducing-halloumi}
}
```
