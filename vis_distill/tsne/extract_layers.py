# coding=utf-8
from transformers import BertForMaskedLM
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extraction some layers")
    parser.add_argument("--model_type", default="bert", choices=["bert"])
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str)
    parser.add_argument("--dump_checkpoint", default='serialization_dir/tf_bert-base-uncased_0247911.pth', type=str)
    parser.add_argument("--vocab_transform", action='store_true')
    args = parser.parse_args()


    if args.model_type == 'bert':
        model = BertForMaskedLM.from_pretrained(args.model_name_or_path)
        prefix = 'bert'
    else:
        raise ValueError(f'args.model_type should be "bert".')

    state_dict = model.state_dict()
    compressed_sd = {}

    for w in ['word_embeddings', 'position_embeddings', 'token_type_embeddings']:
        compressed_sd[f'{prefix}.embeddings.{w}.weight'] = \
            state_dict[f'{prefix}.embeddings.{w}.weight']
    for w in ['weight', 'bias']:
        compressed_sd[f'{prefix}.embeddings.LayerNorm.{w}'] = \
            state_dict[f'{prefix}.embeddings.LayerNorm.{w}']
        compressed_sd[f'{prefix}.pooler.dense.{w}'] = \
            state_dict[f'{prefix}.pooler.dense.{w}']
        
    #with open('exps.txt', 'r', encoding='utf-8') as f:
    #    lines = [line.strip() for line in f]
    lines = ['0-3-8-9', '0-6-8-10', '0-6-8-11', '0-6-9-10', '0-6-9-11']

    for line in lines:
        line = line.strip()
        tidx = [int(_) for _ in line.split('-')]
        print('Teacher Layers:', tidx)    
        std_idx = 0
        for selected_idx in tidx:
            for w in ['weight', 'bias']:
                compressed_sd[f'{prefix}.encoder.layer.{std_idx}.attention.self.query.{w}'] = \
                    state_dict[f'{prefix}.encoder.layer.{selected_idx}.attention.self.query.{w}']
                compressed_sd[f'{prefix}.encoder.layer.{std_idx}.attention.self.key.{w}'] = \
                    state_dict[f'{prefix}.encoder.layer.{selected_idx}.attention.self.key.{w}']
                compressed_sd[f'{prefix}.encoder.layer.{std_idx}.attention.self.value.{w}'] = \
                    state_dict[f'{prefix}.encoder.layer.{selected_idx}.attention.self.value.{w}']

                compressed_sd[f'{prefix}.encoder.layer.{std_idx}.attention.output.dense.{w}'] = \
                    state_dict[f'{prefix}.encoder.layer.{selected_idx}.attention.output.dense.{w}']
                compressed_sd[f'{prefix}.encoder.layer.{std_idx}.attention.output.LayerNorm.{w}'] = \
                    state_dict[f'{prefix}.encoder.layer.{selected_idx}.attention.output.LayerNorm.{w}']

                compressed_sd[f'{prefix}.encoder.layer.{std_idx}.intermediate.dense.{w}'] = \
                    state_dict[f'{prefix}.encoder.layer.{selected_idx}.intermediate.dense.{w}']
                compressed_sd[f'{prefix}.encoder.layer.{std_idx}.output.dense.{w}'] = \
                    state_dict[f'{prefix}.encoder.layer.{selected_idx}.output.dense.{w}']
                compressed_sd[f'{prefix}.encoder.layer.{std_idx}.output.LayerNorm.{w}'] = \
                    state_dict[f'{prefix}.encoder.layer.{selected_idx}.output.LayerNorm.{w}']
            std_idx += 1

        print(f'N layers selected for distillation: {std_idx}')
        print(f'Number of params transfered for distillation: {len(compressed_sd.keys())}')

        args.dump_checkpoint = args.dump_checkpoint.replace('[tidx]', line)
        print(f'Save transfered checkpoint to {args.dump_checkpoint}.')
        torch.save(compressed_sd, args.dump_checkpoint)
        args.dump_checkpoint = args.dump_checkpoint.replace(line, '[tidx]')
        