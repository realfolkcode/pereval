import os
import torch
import argparse

from aria.model import TransformerEMB, ModelConfig
from aria.config import load_model_config
from ariautils.tokenizer import AbsTokenizer
from safetensors.torch import load_file

from pereval.feature_extractor import get_aria_embeddings
from pereval.utils import load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help='The path to aria model checkpoint')
    parser.add_argument('--evaluation_path', type=str, required=False, help='A directory with performances or a .json dictionary with performance paths')
    parser.add_argument('--output_path', type=str, required=False, help='The output path to store embeddings in a .pt format.')

    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    evaluation_path = args.evaluation_path
    output_path = args.output_path

    # Process directory/dictionary input arguments
    evaluation_dir = None
    evaluation_path_dict = None
    if os.path.isdir(evaluation_path):
        evaluation_dir = evaluation_path
    elif evaluation_path.endswith('.json'):
        evaluation_path_dict = load_json(evaluation_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_config = ModelConfig(**load_model_config(name="medium-emb"))
    model_config.set_vocab_size(AbsTokenizer().vocab_size)
    model = TransformerEMB(model_config).to(device)
    state_dict = load_file(filename=checkpoint_path)
    model.load_state_dict(state_dict=state_dict, strict=True)

    emb_dict = get_aria_embeddings(
        model=model,
        device=device,
        evaluation_dir=evaluation_dir,
        evaluation_path_dict=evaluation_path_dict)

    torch.save(emb_dict, output_path)


if __name__ == "__main__":
    main()
