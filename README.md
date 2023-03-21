# BLOOM-fine-tuning

This project is for fine-tuning BLOOM. The repo contains:
- We use [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

Try it on Google Colab! <a href="https://colab.research.google.com/github/hyintell/BLOOM-fine-tuning/blob/main/finetune.ipynb"> 
        <img alt="Build" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>

## Installation

```bash
pip install -r requirements.txt
```

Data: [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

## Training

### alpaca

```bash
python finetune-alpaca.py \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --fp16 True \
    --logging_steps 10 \
    --output_dir output
```

## Progress
- [x] Add Google Colab example
- [x] Test bloom-560m
- [x] Test bloom-1b7
- [ ] Test bloom-7b1
- [ ] Support Evaluation