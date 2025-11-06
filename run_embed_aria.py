import os
import torch
import argparse

from aria.model import TransformerEMB, ModelConfig
from aria.config import load_model_config
from ariautils.tokenizer import AbsTokenizer
from safetensors.torch import load_file

from pereval.feature_extractor import get_aria_embeddings


def main(args):
    checkpoint_path = args.checkpoint_path
    evaluation_dir = args.evaluation_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_config = ModelConfig(**load_model_config(name="medium-emb"))
    model_config.set_vocab_size(AbsTokenizer().vocab_size)
    model = TransformerEMB(model_config).to(device)
    state_dict = load_file(filename=checkpoint_path)
    model.load_state_dict(state_dict=state_dict, strict=True)

    emb_dict = get_aria_embeddings(evaluation_dir,
                                   model,
                                   device)

    emb_path = os.path.join(evaluation_dir, "aria_emb.pt")
    torch.save(emb_dict, emb_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help='path to aria model checkpoint')
    parser.add_argument('--evaluation_dir', type=str, required=True, help='path to evaluation dir')

    args = parser.parse_args()

    main(args)
