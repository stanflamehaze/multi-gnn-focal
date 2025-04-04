# import torch
# import tqdm
# from sklearn.metrics import f1_score
# from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero, save_model, load_model
# from models import GINe, PNA, GATe, RGCN
# from torch_geometric.data import Data, HeteroData
# from torch_geometric.nn import to_hetero, summary
# from torch_geometric.utils import degree
# import wandb
# import logging
# from train_util import FocalLoss # Add this line


import torch
import tqdm
# import wandb  # Make sure wandb is imported
# import logging
import json   # Needed for extract_param if using json

from sklearn.metrics import f1_score
# Import necessary functions and classes from your other files
from train_util import (
    AddEgoIds, extract_param, add_arange_ids, get_loaders,
    evaluate_homo, evaluate_hetero, save_model, load_model,
    FocalLoss  # <-- Make sure FocalLoss is imported from train_util
)
from models import GINe, PNA, GATe, RGCN
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
import wandb
import logging

def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    best_val_f1 = 0
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            #remove the unique edge id from the edge features, as it's no longer needed
            batch.edge_attr = batch.edge_attr[:, 1:]

            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = out[mask]
            ground_truth = batch.y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(ground_truth)
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        wandb.log({"f1/train": f1}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}')

        #evaluate
        val_f1 = evaluate_homo(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_homo(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"f1/test": te_f1}, step=epoch)
        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Test F1: {te_f1:.4f}')

        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
    
    return model

def train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    best_val_f1 = 0
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)
            
            #remove the unique edge id from the edge features, as it's no longer needed
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            out = out[('node', 'to', 'node')]
            pred = out[mask]
            ground_truth = batch['node', 'to', 'node'].y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            
        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        wandb.log({"f1/train": f1}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}')

        #evaluate
        val_f1 = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"f1/test": te_f1}, step=epoch)
        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Test F1: {te_f1:.4f}')

        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
        
    return model

def get_model(sample_batch, config, args):
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    if args.model == "gin":
        model = GINe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), residual=False, edge_updates=args.emlps, edge_dim=e_dim, 
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "gat":
        model = GATe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), n_heads=round(config.n_heads), 
                edge_updates=args.emlps, edge_dim=e_dim,
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "pna":
        if not isinstance(sample_batch, HeteroData):
            d = degree(sample_batch.edge_index[1], dtype=torch.long)
        else:
            index = torch.cat((sample_batch['node', 'to', 'node'].edge_index[1], sample_batch['node', 'rev_to', 'node'].edge_index[1]), 0)
            d = degree(index, dtype=torch.long)
        deg = torch.bincount(d, minlength=1)
        model = PNA(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, deg=deg, final_dropout=config.final_dropout
            )
    elif config.model == "rgcn":
        model = RGCN(
            num_features=n_feats, edge_dim=e_dim, num_relations=8, num_gnn_layers=round(config.n_gnn_layers),
            n_classes=2, n_hidden=round(config.n_hidden),
            edge_update=args.emlps, dropout=config.dropout, final_dropout=config.final_dropout, n_bases=None #(maybe)
        )
    
    return model

# def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
#     #set device
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     #define a model config dictionary and wandb logging at the same time
#     wandb.init(
#         mode="disabled" if args.testing else "online",
#         project="your_proj_name", #replace this with your wandb project name if you want to use wandb logging

#         config={
#             "epochs": args.n_epochs,
#             "batch_size": args.batch_size,
#             "model": args.model,
#             "data": args.data,
#             "num_neighbors": args.num_neighs,
#             "lr": extract_param("lr", args),
#             "n_hidden": extract_param("n_hidden", args),
#             "n_gnn_layers": extract_param("n_gnn_layers", args),
#             "loss": "ce",
#             "w_ce1": extract_param("w_ce1", args),
#             "w_ce2": extract_param("w_ce2", args),
#             "dropout": extract_param("dropout", args),
#             "final_dropout": extract_param("final_dropout", args),
#             "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
#         }
        
#     )

#     config = wandb.config



#     #set the transform if ego ids should be used
#     if args.ego:
#         transform = AddEgoIds()
#     else:
#         transform = None

#     #add the unique ids to later find the seed edges
#     add_arange_ids([tr_data, val_data, te_data])

#     tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

#     #get the model
#     sample_batch = next(iter(tr_loader))
#     model = get_model(sample_batch, config, args)

#     if args.reverse_mp:
#         model = to_hetero(model, te_data.metadata(), aggr='mean')
    
#     if args.finetune:
#         model, optimizer = load_model(model, device, args, config, data_config)
#     else:
#         model.to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
#     sample_batch.to(device)
#     sample_x = sample_batch.x if not isinstance(sample_batch, HeteroData) else sample_batch.x_dict
#     sample_edge_index = sample_batch.edge_index if not isinstance(sample_batch, HeteroData) else sample_batch.edge_index_dict
#     if isinstance(sample_batch, HeteroData):
#         sample_batch['node', 'to', 'node'].edge_attr = sample_batch['node', 'to', 'node'].edge_attr[:, 1:]
#         sample_batch['node', 'rev_to', 'node'].edge_attr = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
#     else:
#         sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
#     sample_edge_attr = sample_batch.edge_attr if not isinstance(sample_batch, HeteroData) else sample_batch.edge_attr_dict
#     logging.info(summary(model, sample_x, sample_edge_index, sample_edge_attr))
    
#     loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))

#     if args.reverse_mp:
#         model = train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
#     else:
#         model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    
#     wandb.finish()

# =============================================
# Updated train_gnn function
# =============================================
def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Dynamically build configuration for wandb ---
    run_config = {
        "epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "model": args.model,
        "data": args.data,
        "num_neighbors": args.num_neighs,
        "lr": extract_param("lr", args),
        "n_hidden": extract_param("n_hidden", args),
        "n_gnn_layers": extract_param("n_gnn_layers", args),
        "dropout": extract_param("dropout", args),
        "final_dropout": extract_param("final_dropout", args),
        "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        # Add any other general parameters extracted here
    }

    # Extract the loss type from the settings file
    loss_type = extract_param("loss", args)
    if loss_type is None:
        logging.warning(f"Loss type not found in settings for model {args.model}, defaulting to 'ce'.")
        loss_type = "ce" # Default or raise error
    run_config["loss"] = loss_type
    logging.info(f"Extracted loss type: {loss_type}")

    # Add loss-specific parameters to the config
    if loss_type == "focal":
        run_config["alpha"] = extract_param("alpha", args)
        run_config["gamma"] = extract_param("gamma", args)
        # Provide a default gamma if not specified in settings
        if run_config["gamma"] is None:
            logging.warning("Gamma not found for focal loss, defaulting to 2.0.")
            run_config["gamma"] = 2.0
        # Alpha can be None (no weighting), float, or list - handled by FocalLoss class
        logging.info(f"Focal loss params: alpha={run_config['alpha']}, gamma={run_config['gamma']}")

    elif loss_type == "ce":
        run_config["w_ce1"] = extract_param("w_ce1", args)
        run_config["w_ce2"] = extract_param("w_ce2", args)
        # Provide default weights if not specified
        if run_config["w_ce1"] is None:
            logging.warning("w_ce1 not found for CE loss, defaulting to 1.0.")
            run_config["w_ce1"] = 1.0
        if run_config["w_ce2"] is None:
            logging.warning("w_ce2 not found for CE loss, defaulting to 1.0.")
            run_config["w_ce2"] = 1.0
        logging.info(f"CE loss params: weights=[{run_config['w_ce1']}, {run_config['w_ce2']}]")
    else:
        logging.error(f"Unsupported loss type '{loss_type}' configured for model {args.model}.")
        raise ValueError(f"Unsupported loss type: {loss_type}")

    # Initialize wandb with the dynamically created config
    wandb.init(
        mode="disabled" if args.testing else "online",
        project="your_proj_name", # replace this with your wandb project name
        config=run_config
    )
    config = wandb.config # Use the config object from wandb

    # set the transform if ego ids should be used
    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    # add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    # get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args) # Pass the wandb config

    if args.reverse_mp:
        # Ensure metadata is correct if using reverse MP with hetero data
        try:
            metadata = te_data.metadata()
            model = to_hetero(model, metadata, aggr='mean')
            logging.info("Converted model to heterogeneous.")
        except Exception as e:
            logging.error(f"Failed to convert model to heterogeneous: {e}")
            # Handle error appropriately, maybe exit or proceed with caution
            raise e

    # Load or initialize optimizer
    if args.finetune:
        try:
            model, optimizer = load_model(model, device, args, config, data_config)
            logging.info(f"Loaded model and optimizer for fine-tuning from checkpoint {args.unique_name}.")
        except FileNotFoundError:
             logging.error(f"Finetune checkpoint not found: {data_config['paths']['model_to_load']}/checkpoint_{args.unique_name}.tar")
             raise
    else:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        logging.info(f"Initialized new model and optimizer with lr={config.lr}.")

    # Log model summary
    try:
        # Prepare sample data for summary (ensure it's on the right device)
        sample_batch.to(device)
        sample_x = sample_batch.x if not isinstance(sample_batch, HeteroData) else sample_batch.x_dict
        sample_edge_index = sample_batch.edge_index if not isinstance(sample_batch, HeteroData) else sample_batch.edge_index_dict

        # Create temporary sample edge attributes without the ID column
        if isinstance(sample_batch, HeteroData):
             sample_edge_attr_dict = {}
             for edge_type, edge_store in sample_batch.edge_items():
                 if 'edge_attr' in edge_store and edge_store.edge_attr is not None:
                     sample_edge_attr_dict[edge_type] = edge_store.edge_attr[:, 1:] # Remove ID column
                 else:
                      sample_edge_attr_dict[edge_type] = None # Or handle as needed if no edge_attr
             sample_edge_attr = sample_edge_attr_dict
        else:
            if sample_batch.edge_attr is not None:
                sample_edge_attr = sample_batch.edge_attr[:, 1:] # Remove ID column
            else:
                sample_edge_attr = None

        logging.info("Model Summary:\n" + str(summary(model, sample_x, sample_edge_index, sample_edge_attr)))
    except Exception as e:
        logging.error(f"Could not generate model summary: {e}")
        # Log structure details manually if summary fails
        logging.info(f"Model: {model}")
        if isinstance(sample_batch, HeteroData):
            logging.info(f"Sample Batch Metadata: {sample_batch.metadata()}")
            logging.info(f"Sample x_dict keys: {sample_batch.x_dict.keys()}")
            logging.info(f"Sample edge_index_dict keys: {sample_batch.edge_index_dict.keys()}")
        else:
             logging.info(f"Sample Batch Keys: {sample_batch.keys}")


    # --- Instantiate the loss function based on config ---
    if config.loss == "focal":
        loss_fn = FocalLoss(alpha=config.alpha, gamma=config.gamma, reduction='mean').to(device)
        logging.info(f"Instantiated Focal Loss (alpha={config.alpha}, gamma={config.gamma})")
    elif config.loss == "ce":
        ce_weights = torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=ce_weights).to(device)
        logging.info(f"Instantiated Cross Entropy Loss (weights=[{config.w_ce1}, {config.w_ce2}])")
    else:
        # This case should ideally be caught earlier, but added for safety
        logging.error(f"Trying to instantiate unsupported loss: {config.loss}")
        raise ValueError(f"Unsupported loss type during instantiation: {config.loss}")
    # --- Loss function instantiated ---

    # Call the appropriate training loop
    if args.reverse_mp:
         logging.info("Starting training with HeteroData (reverse MP enabled)...")
         model = train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    else:
         logging.info("Starting training with Homogeneous Data...")
         model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)

    logging.info("Training finished.")
    wandb.finish()
    logging.info("Wandb run finished.")

# =============================================
# End of updated train_gnn function
# =============================================