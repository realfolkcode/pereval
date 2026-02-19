import os
import torch
import argparse

from aria.model import TransformerEMB, TransformerLM, ModelConfig
from aria.config import load_model_config
from ariautils.tokenizer import AbsTokenizer
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from pereval.feature_extractor import get_full_embedding_from_midi


def main(args):
    checkpoint_path = args.checkpoint_path
    evaluation_dir = args.evaluation_dir
    output_dir = args.output_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    #model_config = ModelConfig(**load_model_config(name="medium-emb"))
    model_config = ModelConfig(**load_model_config(name="medium-emb"))
    model_config.set_vocab_size(AbsTokenizer().vocab_size)
    #model = TransformerEMB(model_config).to(device)
    model = TransformerLM(model_config).to(device)
    state_dict = load_file(filename=checkpoint_path)
    model.load_state_dict(state_dict=state_dict, strict=True)

    for i, score_name in enumerate(tqdm(os.listdir(evaluation_dir))):
        score_dir = os.path.join(evaluation_dir, score_name)
        if not os.path.isdir(score_dir):
            continue

        score_output_dir = os.path.join(output_dir, score_name)
        os.makedirs(score_output_dir, exist_ok=True)

        for perf_name in os.listdir(score_dir):
            if perf_name.endswith(".mid") or perf_name.endswith(".midi"):
                perf_path = os.path.join(score_dir, perf_name)
                emb = get_full_embedding_from_midi(
                    model=model,
                    midi_path=perf_path,
                    device=device,
                    is_contrastive=True
                ).cpu()

                output_filename = perf_name.split('.')[0]
                output_filename = f"{output_filename}.pt"
                output_path = os.path.join(score_output_dir, output_filename)
                torch.save(emb, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help='path to aria model checkpoint')
    parser.add_argument('--evaluation_dir', type=str, required=True, help='path to evaluation dir')
    parser.add_argument('--output_dir', type=str, required=True, help='The path to the output directory')

    args = parser.parse_args()

    main(args)
