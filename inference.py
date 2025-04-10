import torch
import pandas as pd
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero
from training import get_model
from torch_geometric.nn import to_hetero, summary
import wandb
import logging
import os
import sys
import time

script_start = time.time()

def infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #define a model config dictionary and wandb logging at the same time
    wandb.init(
        mode="disabled" if args.testing else "online",
        project="your_proj_name",

        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        }
    )

    config = wandb.config

    #set the transform if ego ids should be used
    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    #get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')
    
    if not (args.avg_tps or args.finetune):
        command = " ".join(sys.argv)
        name = ""
        name = '-'.join(name.split('-')[3:])
        args.unique_name = name

    logging.info("=> loading model checkpoint")
    checkpoint = torch.load(f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict']) # [Source 50]
    model.to(device) # [Source 51]
    model.eval() # 确保在推理前设置为评估模式

    logging.info("=> loaded checkpoint (epoch {}) for inference".format(start_epoch)) # [Source 51]

    # 调用更新后的评估函数
    # 注意：原始代码中传递了 precrec=True 参数，但在 train_util.py 的函数定义中似乎未使用。
    # 由于我们现在总是计算 precision 和 recall，可以移除这个参数。
    if not args.reverse_mp:
        te_f1, te_precision, te_recall, te_conf_matrix = evaluate_homo(te_loader, te_inds, model, te_data, device, args) # [Source 51] 更新调用
    else:
        te_f1, te_precision, te_recall, te_conf_matrix = evaluate_hetero(te_loader, te_inds, model, te_data, device, args) # [Source 51] 更新调用

    # 记录最终的推理指标
    logging.info(f"--- Final Inference Results ---")
    logging.info(f"  Test F1: {te_f1:.4f}")
    logging.info(f"  Test Precision: {te_precision:.4f}")
    logging.info(f"  Test Recall: {te_recall:.4f}")
    logging.info(f"  Test Confusion Matrix:\n{te_conf_matrix}")

    # (可选) 将结果记录到 wandb
    wandb.log({
        "inference/test_f1": te_f1,
        "inference/test_precision": te_precision,
        "inference/test_recall": te_recall
    })
    # wandb 不直接支持记录 numpy 数组（混淆矩阵），可以将其转换为字符串或记录 TP, TN, FP, FN
    tn, fp, fn, tp = te_conf_matrix.ravel()
    wandb.log({
        "inference/test_TP": tp,
        "inference/test_TN": tn,
        "inference/test_FP": fp,
        "inference/test_FN": fn
    })


    wandb.finish() 
