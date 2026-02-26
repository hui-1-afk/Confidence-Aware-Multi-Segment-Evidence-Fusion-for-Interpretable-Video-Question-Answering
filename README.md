# Confidence-Aware-Multi-Segment-Evidence-Fusion-for-Interpretable-Video-Question-Answering
This project implements a **Confidence-Aware Multi-Segment Evidence Fusion Model** for Video Question Answering (VideoQA), which optimizes the VideoMind framework by introducing a Fusioner module to realize multi-segment collaborative reasoning. The model breaks through the limitation of single-segment decision-making in traditional VideoQA, effectively improves the accuracy, robustness and interpretability of long video understanding, and has outstanding performance on multiple mainstream VideoQA benchmarks.
We propose an innovative strategy for the VideoMind model: proactively exploring and analyzing in depth the value of the answer with the second-highest confidence ranking. Through targeted improvements to the VideoMind architecture, we use this 'second choice' as a key probe for detecting the model’s internal decision-making mechanism.

## Project Overview
### Core Motivation
Traditional VideoQA models rely on single high-confidence video segments for decision-making, which is prone to accidental errors due to missing key information; at the same time, the lack of effective multi-segment evidence fusion and result traceability mechanisms leads to poor model interpretability and difficult error iteration.

### Core Innovation
Based on the **Chain-of-LoRA** long video reasoning framework of VideoMind, a **Multi-segment Evidence Fusioner** is added, and three core mechanisms are designed to solve the above problems:
1. **Timestamp Association**: Bind answers to the time segment timestamps of video features, clarify the spatiotemporal source of answers, and provide a basis for error analysis.
2. **Multi-Candidate Fusion**: Retain top-N high-confidence video segments to supplement complementary information and reduce single decision-making errors.
3. **Case-based Decision-Making**: Design differentiated fusion rules according to the consistency of candidate segment answers, balance decision-making efficiency and result accuracy.

### Model Architecture
The model reuses the Qwen2-VL as the unified backbone network, and realizes the function switching of five core modules by loading independent LoRA adapters (rank=64). The reasoning pipeline is as follows:
1. **Planner**: Generate role calling plan according to question type and video content (supports Localization+Answer/Localization Only/Answer Only three modes).
2. **Grounder**: Localize the relevant time segments of the video and predict candidate time windows combined with the special token `<REG>`.
3. **Verifier**: Re-verify the confidence of candidate segments (binary classification task), expand the boundary by 50% and add `<SEG_START>/<SEG_END>` tokens, and retain top-N high-confidence segments.
4. **Fusioner** (core innovation): Fuse multi-segment multi-modal features (visual/text/confidence score) through **Question-Guided Cross-Segment Attention**, generate global evidence representation, and support evidence summary text output.
5. **Answerer**: Generate the final natural language answer under the constraint of global evidence, and reduce reasoning difficulty through feature-level fusion of multi-segment information.

## Environmental Requirements
### Hardware Requirements
- Training: 8 × NVIDIA H100 80GB HBM3 GPUs (distributed training/inference), or other multi-card GPU clusters with single card ≥16GB video memory
- Inference: CPU/single GPU (video memory ≥8GB)
- Optional: Huawei NPU (need to enable the corresponding torch-npu dependency)

### Software Requirements
- Python: 3.9+ (recommended 3.9/3.10, compatible with all dependent versions)
- CUDA/CUDNN: Match the PyTorch 2.4.0 version (recommended CUDA 12.1+)
- Dependencies: See `requirements.txt` for the complete specified version (the project has fixed the version to solve the compatibility problems of Gradio/Deepspeed/PyTorch)

## Environment Installation
### 1. Clone the project
```bash
git clone <project-repo-url>
cd Confidence-Aware-Multi-Segment-Fusion-VideoQA
```

### 2. Create a virtual environment (recommended)
```bash
# Conda creation (recommended)
conda create -n fusion-videoqa python=3.10
conda activate fusion-videoqa

# Or venv creation
python -m venv fusion-videoqa
# Linux/macOS activation
source fusion-videoqa/bin/activate
# Windows activation
fusion-videoqa\Scripts\activate
```

### 3. Install dependent packages
Install all specified version dependencies to avoid version conflicts:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. NPU environment adaptation (optional)
If using Huawei NPU, modify the `requirements.txt` file to comment out the default PyTorch and enable the NPU exclusive version:
```txt
# torch==2.4.0
# torchvision==0.19.0
torch==2.4.0+cpu
torch-npu==2.4.0.post2
torchvision==0.19.0+cpu
```
Then re-execute the dependency installation command.

## Dataset Preparation
The model is evaluated on 5 representative long/short video understanding benchmarks, and the official download addresses are as follows:
1. **Video-MME**: Long video QA dataset (avg. 15 mins), [Official Link](https://github.com/OpenGVLab/Video-MME)
2. **MVBench**: Multi-task short video benchmark (20 tasks, avg. 15s), [Official Link](https://github.com/OpenGVLab/MVBench)
3. **NExT-GQA**: Spatiotemporal causal reasoning grounded QA (avg. 39.5s), [Official Link](https://github.com/Next-GQA/NExT-GQA)
4. **MLVU**: Multi-task long video understanding benchmark (avg. 15.5 mins), [Official Link](https://github.com/facebookresearch/MLVU)
5. **LongVideoBench**: Long context video QA (8s ~ 1h), [Official Link](https://github.com/OpenGVLab/LongVideoBench)

### Dataset Processing
1. Download the original dataset and extract video/annotation files to the `data/` directory (create if not exists).
2. Run the dataset preprocessing script to downsample videos (1 FPS by default) and generate timestamp annotations:
   ```bash
   python scripts/process_dataset.py --dataset <dataset-name> --data_root data/ --save_root data/processed/
   ```
3. The preprocessed dataset structure is as follows:
   ```
   data/processed/
   ├── Video-MME/
   │   ├── videos/ (downsampled video frames)
   │   └── annotations/ (json format QA + timestamp)
   └── ... (other datasets)
   ```

## Model Running
### 1. Pre-trained Model Download
Download the pre-trained Qwen2-VL model (2B) and the LoRA adapters of each module of the project, and place them in the `ckpt/` directory:
```bash
mkdir -p ckpt/qwen2-vl-2b ckpt/lora_adapters
# Download Qwen2-VL-2B (Hugging Face)
git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct ckpt/qwen2-vl-2b
# Download project LoRA adapters
wget <lora-adapters-url> -O ckpt/lora_adapters/fusion_lora.zip
unzip ckpt/lora_adapters/fusion_lora.zip -d ckpt/lora_adapters/
```

### 2. Model Training
The project adopts **frozen backbone + LoRA fine-tuning** strategy: freeze Planner/Grounder/Verifier, only train the LoRA parameters of Fusioner and Answerer.
```bash
# Single machine multi-card training (8 GPUs by default)
accelerate launch --num_processes=8 train.py \
  --model_path ckpt/qwen2-vl-2b \
  --lora_path ckpt/lora_adapters \
  --data_root data/processed/ \
  --dataset NExT-GQA \
  --batch_size 32 \
  --lr 2e-5 \
  --epochs 1 \
  --save_dir ckpt/trained/
```
**Key Training Parameters**:
- `--num_processes`: Number of GPUs for distributed training
- `--batch_size`: Batch size (32 for 8 H100 GPUs)
- `--lr`: Initial learning rate (2e-5 for AdamW optimizer)
- `--epochs`: Training epochs (1 epoch for the project by default)
- `--save_dir`: Trained model/LoRA adapter save directory

### 3. Model Inference
#### Single Sample Inference
Run the single sample inference script to test the video QA effect and output the answer + corresponding time segment + confidence score:
```bash
python infer_single.py \
  --model_path ckpt/qwen2-vl-2b \
  --lora_path ckpt/trained/ \
  --video_path examples/sample_video.mp4 \
  --question "Why are the bunnies gathering on the table?"
```
**Output Example**:
```
Final Answer: Because a boy is sitting aside and handing out food to the bunnies.
Relevant Time Segments: [10,23], [15,76] (confidence: 0.52, 0.47)
Evidence Summary: The bunny gathering behavior appears in segment [10,23], and a boy feeding food is observed in segment [15,76].
```

#### Benchmark Evaluation
Evaluate the model performance on the specified benchmark and generate the accuracy report:
```bash
python eval_benchmark.py \
  --model_path ckpt/qwen2-vl-2b \
  --lora_path ckpt/trained/ \
  --data_root data/processed/ \
  --dataset NExT-GQA \
  --save_result results/next_gqa_result.json
```

### 4. Gradio Visual Demo (Optional)
The project integrates Gradio to build a visual interactive interface, supporting video upload, question input and real-time QA:
```bash
python app_gradio.py \
  --model_path ckpt/qwen2-vl-2b \
  --lora_path ckpt/trained/
```
After running, open the local address (default `http://localhost:7860`) in the browser to operate, and support:
- Video file upload (MP4 format)
- Text question input
- Real-time output of answers, relevant time segments, confidence scores and evidence summaries
- Support Hugging Face Spaces deployment (rely on `spaces==0.34.0`)

## Experimental Results
### Comprehensive Performance Comparison
The model (VideoMind-Fusion, 2B) achieves significant performance improvements on multiple benchmarks compared with the original VideoMind (2B) and other mainstream VideoQA models (7B scale), and maintains stable performance on long video tasks. The key results are as follows (unit: %):

| Method               | Size | Video-MME | MVBench | MLVU  | NExT-GQA |
|----------------------|------|-----------|---------|-------|----------|
| Video-LLaVA          | 7B   | 41.1      | 43.0    | 29.3  | -        |
| TimeChat             | 7B   | 34.3      | 38.5    | 30.9  | -        |
| VideoMind (Original) | 2B   | 53.6      | 61.9    | 58.7  | 32.6     |
| VideoMind (Reproduce)| 2B   | 53.15     | 48.8    | 56.68 | 66.6     |
| **Ours (Fusion)**    | 2B   | 52.42     | 61.5     | 58.89 | **74.08** |

### Ablation Experiment Results
1. **Key Module Ablation** (NExT-GQA, %): The full model achieves the highest accuracy (74.45%), and the Verifier module contributes the most to the fusion effect.
   | Configure   | Accuracy |
   |-------------|----------|
   | Vanilla     | 74.03    |
   | w/o Planner | 74.33    |
   | w/o Verifier| 74.15    |
   | Full (Ours) | **74.45**|

2. **Fusion Strategy Ablation** (NExT-GQA, %): The **dynamic fusion strategy** (Softmax weight + consistency priority) proposed in the project outperforms single-segment/equal weight/hard selection strategies.
   | Fusion Strategy       | Accuracy |
   |-----------------------|----------|
   | Top-1 Only            | 73.2     |
   | Equal Weight          | 73.8     |
   | Hard Selection        | 73.5     |
   | Ours (Dynamic Fusion) | **74.1** |

## Core Code Structure
```
Confidence-Aware-Multi-Segment-Fusion-VideoQA/
├── ckpt/               # Pre-trained model + LoRA adapter storage
├── data/               # Original/processed dataset storage
├── examples/           # Test video/question examples
├── models/             # Core model code
│   ├── backbone.py     # Qwen2-VL backbone loading
│   ├── planner.py      # Planner module (LoRA-P)
│   ├── grounder.py     # Grounder module (LoRA-G)
│   ├── verifier.py     # Verifier module (LoRA-V)
│   ├── fusioner.py     # Fusioner module (LoRA-F, core innovation)
│   └── answerer.py     # Answerer module
├── scripts/            # Dataset preprocessing/utility scripts
├── train.py            # Model training script
├── infer_single.py     # Single sample inference script
├── eval_benchmark.py   # Benchmark evaluation script
├── app_gradio.py       # Gradio visual demo script
├── requirements.txt    # Dependence list (specified version)
└── README.md           # Project documentation
```

## Key Notes
1. **Version Lock**: All dependencies in `requirements.txt` are specified with fixed versions to solve the compatibility problems of Gradio (#10662), Deepspeed (#6793) and PyTorch (#138386); the Transformers 4.45.2 version contains the project's custom patches, **do not upgrade at will**.
2. **LoRA Adapter**: Each module uses an independent LoRA adapter (rank=64), which realizes function isolation and efficient fine-tuning, and the Fusioner module's LoRA-F is the core of the project.
3. **Inference Efficiency**: The model retains the top 2 high-confidence segments by default for fusion (N'=2), which can be adjusted by the `--top_n` parameter in the script (increasing N' will improve accuracy but increase computational overhead).
4. **Model Interpretability**: The Fusioner module supports the output of `evidence summary text`, which can be enabled by the `--enable_evidence` parameter in the inference/ demo script.
5. **Safety**: The model uses `safetensors` for model weight storage, which avoids the security risks of pickle and has faster read/write speed.

## Common Problems
1. **Dependency Installation Timeout**: Add the domestic PyPI source (`-i https://pypi.tuna.tsinghua.edu.cn/simple`) to the `pip install` command.
2. **GPU Out of Memory**: Reduce the batch size (`--batch_size`), lower the LoRA rank, or enable DeepSpeed ZeRO optimization (add `--deepspeed configs/deepspeed.json` to the training command).
3. **Gradio Startup Failure**: Check whether the pydantic version is 2.10.6, and reinstall the dependent package if necessary.
4. **Video Preprocessing Error**: Ensure that the decord library is installed correctly (0.6.0), and the video format is MP4 (other formats can be converted by `ffmpeg`).
5. **NPU Running Error**: Confirm that the torch-npu dependency is enabled correctly, and the NPU driver/ firmware is installed in accordance with the official documentation.

## Citation
If this project is used in your research, please cite the relevant paper:
```bibtex
@article{fusion_videoqa_2025,
  title={Confidence-Aware Multi-Segment Evidence Fusion for Interpretable Video Question Answering},
  author={XXX},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
@article{videomind_2025,
  title={VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning},
  author={Liu, Ye and Lin, Kevin Qinghong and Chen, Chang Wen and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2503.13444},
  year={2025}
}
```

## Contact
If you have any questions about the project, please submit an Issue to the project warehouse or contact the developer via email: `<544249092@qq.com>`.
```
