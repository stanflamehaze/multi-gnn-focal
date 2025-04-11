import torch
import tqdm
# 确保导入所需的指标函数
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero, save_model, load_model
from models import GINe, PNA, GATe, RGCN
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
import wandb
import logging
import numpy as np # 确保导入 numpy

# Define Focal Loss class
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=alpha, reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    """
    Trains and evaluates a homogenous GNN model.
    """
    best_val_f1 = 0  # 可以根据需要更改为 best_val_recall 等指标
    logging.info(f"Starting homogenous GNN training for {config.epochs} epochs...")
    logging.info(f"Config: {config}") # 打印配置信息

    for epoch in range(config.epochs):
        model.train()  # 设置模型为训练模式
        total_loss = total_examples = 0
        preds_epoch = []
        ground_truths_epoch = []
        logging.info(f"--- Epoch {epoch+1}/{config.epochs} ---")

        # --- Training Loop ---
        for batch in tqdm.tqdm(tr_loader, desc=f"Epoch {epoch+1} Training", disable=not args.tqdm):
            optimizer.zero_grad()
            # select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            # remove the unique edge id from the edge features
            batch.edge_attr = batch.edge_attr[:, 1:] # [Source 61]

            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr) # [Source 61]
            pred = out[mask]
            ground_truth = batch.y[mask]

            # Check if ground_truth is empty (can happen in rare cases with sampling)
            if ground_truth.numel() == 0:
                logging.warning(f"Epoch {epoch+1}, Batch: Skipping batch due to empty ground truth.")
                continue

            preds_epoch.append(pred.argmax(dim=-1)) # [Source 62]
            ground_truths_epoch.append(ground_truth) # [Source 62]
            loss = loss_fn(pred, ground_truth) # [Source 62]

            loss.backward() # [Source 62]
            optimizer.step() # [Source 62]

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel() # [Source 62]

        # --- Training Metrics Calculation ---
        # Ensure there are predictions before calculating metrics
        if not preds_epoch:
             logging.warning(f"Epoch {epoch+1}: No predictions recorded during training.")
             continue # Skip evaluation if training loop didn't produce results

        pred_train = torch.cat(preds_epoch, dim=0).detach().cpu().numpy() # [Source 63]
        ground_truth_train = torch.cat(ground_truths_epoch, dim=0).detach().cpu().numpy() # [Source 63]

        # Handle case where ground truth might have only one class after filtering/sampling
        if len(np.unique(ground_truth_train)) < 2:
            logging.warning(f"Epoch {epoch+1} Train: Only one class present in ground truth. Metrics might be skewed or invalid.")
            # Set metrics to NaN or 0, or skip logging as appropriate
            f1_train, precision_train, recall_train = 0.0, 0.0, 0.0
            conf_matrix_train = np.zeros((2,2)) # Default empty matrix
        else:
            f1_train = f1_score(ground_truth_train, pred_train, zero_division=0) # [Source 63]
            precision_train = precision_score(ground_truth_train, pred_train, zero_division=0)
            recall_train = recall_score(ground_truth_train, pred_train, zero_division=0)
            conf_matrix_train = confusion_matrix(ground_truth_train, pred_train)

        avg_loss_train = total_loss / total_examples if total_examples > 0 else 0

        # --- Logging Training Metrics ---
        wandb.log({
            "loss/train": avg_loss_train,
            "f1/train": f1_train,
            "precision/train": precision_train,
            "recall/train": recall_train
        }, step=epoch) # [Source 63: Existing wandb log]
        logging.info(f'  Training Loss: {avg_loss_train:.4f}')
        logging.info(f'  Train Metrics - F1: {f1_train:.4f}, Precision: {precision_train:.4f}, Recall: {recall_train:.4f}')
        logging.info(f'  Train Confusion Matrix:\n{conf_matrix_train}')

        # --- Evaluation ---
        model.eval()  # Switch to evaluation mode
        logging.info("  Evaluating on Validation Set...")
        val_f1, val_precision, val_recall, val_conf_matrix = evaluate_homo(val_loader, val_inds, model, val_data, device, args) # [Source 63] 更新调用
        logging.info("  Evaluating on Test Set...")
        te_f1, te_precision, te_recall, te_conf_matrix = evaluate_homo(te_loader, te_inds, model, te_data, device, args) # [Source 63] 更新调用

        # --- Logging Validation Metrics ---
        wandb.log({
            "f1/validation": val_f1,
            "precision/validation": val_precision,
            "recall/validation": val_recall
        }, step=epoch) # [Source 63]
        logging.info(f'  Validation Metrics - F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
        logging.info(f'  Validation Confusion Matrix:\n{val_conf_matrix}')

        # --- Logging Test Metrics ---
        wandb.log({
            "f1/test": te_f1,
            "precision/test": te_precision,
            "recall/test": te_recall
        }, step=epoch) # [Source 64]
        logging.info(f'  Test Metrics - F1: {te_f1:.4f}, Precision: {te_precision:.4f}, Recall: {te_recall:.4f}')
        logging.info(f'  Test Confusion Matrix:\n{te_conf_matrix}')

        # --- Save Best Model Logic ---
        # (Based on validation F1 score, can be changed)
        if epoch == 0:
            best_val_f1 = val_f1 # Initialize best score
            wandb.log({
                "best_test_f1": te_f1,
                "best_test_precision": te_precision,
                "best_test_recall": te_recall
            }, step=epoch) # [Source 64]
            if args.save_model:
                 logging.info(f"  Saving model from epoch {epoch+1} (initial best)")
                 save_model(model, optimizer, epoch, args, data_config) # [Source 65]

        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            logging.info(f"  New best validation F1 found: {best_val_f1:.4f}")
            wandb.log({
                "best_test_f1": te_f1,
                "best_test_precision": te_precision,
                "best_test_recall": te_recall
             }, step=epoch) # [Source 64]
            if args.save_model:
                logging.info(f"  Saving model from epoch {epoch+1}")
                save_model(model, optimizer, epoch, args, data_config) # [Source 65]

    logging.info("Homogenous GNN training finished.")
    return model

def train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    """
    Trains and evaluates a heterogenous GNN model.
    """
    best_val_f1 = 0 # 或根据需要更改为 best_val_recall 等
    logging.info(f"Starting heterogenous GNN training for {config.epochs} epochs...")
    logging.info(f"Config: {config}") # 打印配置信息

    for epoch in range(config.epochs):
        model.train() # 设置模型为训练模式
        total_loss = total_examples = 0
        preds_epoch = []
        ground_truths_epoch = []
        logging.info(f"--- Epoch {epoch+1}/{config.epochs} ---")

        # --- Training Loop ---
        for batch in tqdm.tqdm(tr_loader, desc=f"Epoch {epoch+1} Training", disable=not args.tqdm): # [Source 66]
            optimizer.zero_grad()
            # select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids) # [Source 66-67]

            # remove the unique edge id from the edge features
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:] # [Source 67]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:] # [Source 67]

            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict) # [Source 68]
            out = out[('node', 'to', 'node')] # Select output for the target edge type
            pred = out[mask]
            ground_truth = batch['node', 'to', 'node'].y[mask] # [Source 68]

            # Check if ground_truth is empty
            if ground_truth.numel() == 0:
                logging.warning(f"Epoch {epoch+1}, Batch: Skipping batch due to empty ground truth.")
                continue

            preds_epoch.append(pred.argmax(dim=-1)) # [Source 68]
            ground_truths_epoch.append(ground_truth) # [Source 68]
            loss = loss_fn(pred, ground_truth) # [Source 68]

            loss.backward() # [Source 69]
            optimizer.step() # [Source 69]

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel() # [Source 69]

        # --- Training Metrics Calculation ---
        if not preds_epoch:
             logging.warning(f"Epoch {epoch+1}: No predictions recorded during training.")
             continue

        pred_train = torch.cat(preds_epoch, dim=0).detach().cpu().numpy() # [Source 69]
        ground_truth_train = torch.cat(ground_truths_epoch, dim=0).detach().cpu().numpy() # [Source 69]

        if len(np.unique(ground_truth_train)) < 2:
            logging.warning(f"Epoch {epoch+1} Train: Only one class present in ground truth. Metrics might be skewed or invalid.")
            f1_train, precision_train, recall_train = 0.0, 0.0, 0.0
            conf_matrix_train = np.zeros((2,2))
        else:
            f1_train = f1_score(ground_truth_train, pred_train, zero_division=0) # [Source 69]
            precision_train = precision_score(ground_truth_train, pred_train, zero_division=0)
            recall_train = recall_score(ground_truth_train, pred_train, zero_division=0)
            conf_matrix_train = confusion_matrix(ground_truth_train, pred_train)

        avg_loss_train = total_loss / total_examples if total_examples > 0 else 0

        # --- Logging Training Metrics ---
        wandb.log({
            "loss/train": avg_loss_train,
            "f1/train": f1_train,
            "precision/train": precision_train,
            "recall/train": recall_train
        }, step=epoch) # [Source 70]
        logging.info(f'  Training Loss: {avg_loss_train:.4f}')
        logging.info(f'  Train Metrics - F1: {f1_train:.4f}, Precision: {precision_train:.4f}, Recall: {recall_train:.4f}')
        logging.info(f'  Train Confusion Matrix:\n{conf_matrix_train}')

        # --- Evaluation ---
        model.eval() # Switch to evaluation mode
        logging.info("  Evaluating on Validation Set...")
        val_f1, val_precision, val_recall, val_conf_matrix = evaluate_hetero(val_loader, val_inds, model, val_data, device, args) # [Source 70] 更新调用
        logging.info("  Evaluating on Test Set...")
        te_f1, te_precision, te_recall, te_conf_matrix = evaluate_hetero(te_loader, te_inds, model, te_data, device, args) # [Source 70] 更新调用

        # --- Logging Validation Metrics ---
        wandb.log({
            "f1/validation": val_f1,
            "precision/validation": val_precision,
            "recall/validation": val_recall
        }, step=epoch) # [Source 70]
        logging.info(f'  Validation Metrics - F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
        logging.info(f'  Validation Confusion Matrix:\n{val_conf_matrix}')

        # --- Logging Test Metrics ---
        wandb.log({
            "f1/test": te_f1,
            "precision/test": te_precision,
            "recall/test": te_recall
        }, step=epoch) # [Source 70]
        logging.info(f'  Test Metrics - F1: {te_f1:.4f}, Precision: {te_precision:.4f}, Recall: {te_recall:.4f}')
        logging.info(f'  Test Confusion Matrix:\n{te_conf_matrix}')

        # --- Save Best Model Logic ---
        if epoch == 0:
            best_val_f1 = val_f1 # Initialize best score
            wandb.log({
                "best_test_f1": te_f1,
                "best_test_precision": te_precision,
                "best_test_recall": te_recall
             }, step=epoch) # [Source 71]
            if args.save_model:
                 logging.info(f"  Saving model from epoch {epoch+1} (initial best)")
                 save_model(model, optimizer, epoch, args, data_config) # [Source 71]

        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            logging.info(f"  New best validation F1 found: {best_val_f1:.4f}")
            wandb.log({
                "best_test_f1": te_f1,
                "best_test_precision": te_precision,
                "best_test_recall": te_recall
            }, step=epoch) # [Source 71]
            if args.save_model:
                logging.info(f"  Saving model from epoch {epoch+1}")
                save_model(model, optimizer, epoch, args, data_config) # [Source 71]

    logging.info("Heterogenous GNN training finished.")
    return model # [Source 72]

def get_model(sample_batch, config, args):
    """
    Initializes the GNN model based on configuration.
    """
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    # Correctly calculate edge dimension (excluding the added arange ID)
    e_dim = (sample_batch.edge_attr.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1) # [Source 72]

    logging.info(f"Node features: {n_feats}, Edge features: {e_dim}")

    model = None # Initialize model variable
    if args.model == "gin": # [Source 72]
        model = GINe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), residual=False, edge_updates=args.emlps, edge_dim=e_dim,
                dropout=config.dropout, final_dropout=config.final_dropout # [Source 72-73]
                )
    elif args.model == "gat": # [Source 73]
        model = GATe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), n_heads=round(config.n_heads),
                edge_updates=args.emlps, edge_dim=e_dim,
                dropout=config.dropout, final_dropout=config.final_dropout # [Source 73-74]
                )
    elif args.model == "pna": # [Source 74]
        if not isinstance(sample_batch, HeteroData):
            d = degree(sample_batch.edge_index[1], dtype=torch.long) # [Source 74]
        else:
            index = torch.cat((sample_batch['node', 'to', 'node'].edge_index[1], sample_batch['node', 'rev_to', 'node'].edge_index[1]), 0) # [Source 74]
            d = degree(index, dtype=torch.long) # [Source 75]
        deg = torch.bincount(d, minlength=1) # [Source 75]
        logging.info(f"Node degrees for PNA: {deg[:10]}...") # Log first few degrees
        model = PNA(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, deg=deg, final_dropout=config.final_dropout # [Source 75]
            )
    elif args.model == "rgcn": # Corrected from config.model to args.model [Source 75]
        # RGCN expects edge_type as the last feature if not provided separately.
        # Ensure edge_dim passed here excludes the type if it's handled by RGCNConv internally.
        # Assuming RGCNConv handles relation type via edge_type argument, edge_dim is other features.
         model = RGCN(
             num_features=n_feats, edge_dim=e_dim, num_relations=8, # Assuming 8 relation types based on model file [Source 76]
             num_gnn_layers=round(config.n_gnn_layers),
             n_classes=2, n_hidden=round(config.n_hidden),
             edge_update=args.emlps, dropout=config.dropout, final_dropout=config.final_dropout, n_bases=None # [Source 76]
         )
    else:
         raise ValueError(f"Unknown model type: {args.model}")

    return model

def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    """
    Main function to set up and initiate GNN training.
    """
    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 检查是否有loss类型参数
    loss_type = getattr(args, 'loss', 'ce')  # 默认为'ce'
    
    # 设置focal loss参数
    alpha = getattr(args, 'alpha', [0.75, 0.25])  # 默认值
    gamma = getattr(args, 'gamma', 2.0)  # 默认值
    
    # 日志记录focal loss参数
    if loss_type == 'focal':
        logging.info(f"Extracted loss type: {loss_type}")
        logging.info(f"Focal loss params: alpha={alpha}, gamma={gamma}")
    
    # 定义模型配置字典和wandb日志
    wandb_mode = "disabled" if args.testing else "online"
    project_name = "aml_gnn_project"
    wandb.init(
        mode=wandb_mode,
        project=project_name,
        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": loss_type,  # 使用动态loss类型
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None,
            # 添加focal loss参数
            "alpha": alpha if loss_type == 'focal' else None,
            "gamma": gamma if loss_type == 'focal' else None,
            # 其他适配
            "emlps": args.emlps,
            "reverse_mp": args.reverse_mp,
            "ports": args.ports,
            "tds": args.tds,
            "ego": args.ego,
            "seed": args.seed
        }
    )
    # Check if wandb is online and log run name/ID
    if wandb.run is not None:
        logging.info(f"Wandb run initialized: {wandb.run.name} (ID: {wandb.run.id})")
        logging.info(f"Wandb dashboard: {wandb.run.get_url()}")

    config = wandb.config # [Source 79]

    #set the transform if ego ids should be used
    if args.ego: # [Source 79]
        transform = AddEgoIds()
        logging.info("Using AddEgoIds transform.")
    else:
        transform = None

    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data]) # [Source 80]

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args) # [Source 80]

    #get the model
    # Need a sample batch to determine input/output sizes correctly
    try:
        sample_batch = next(iter(tr_loader)) # [Source 80]
    except StopIteration:
        logging.error("Training loader is empty! Cannot initialize model.")
        return

    model = get_model(sample_batch, config, args) # [Source 80]

    # Convert model to heterogenous if reverse MP is enabled
    if args.reverse_mp: # [Source 80]
        if not isinstance(tr_data, HeteroData):
             logging.error("Reverse MP selected, but data is not HeteroData. Aborting.")
             return
        logging.info("Converting model to heterogeneous.")
        model = to_hetero(model, tr_data.metadata(), aggr='mean') # Use tr_data metadata

    # Load or initialize model and optimizer
    if args.finetune: # [Source 80]
        logging.info(f"Finetuning model from checkpoint: {args.unique_name}")
        try:
             model, optimizer = load_model(model, device, args, config, data_config) # [Source 80]
        except FileNotFoundError:
             logging.error(f"Checkpoint file not found for finetuning: {data_config['paths']['model_to_load']}/checkpoint_{args.unique_name}.tar")
             return
    else:
        logging.info("Initializing new model parameters and optimizer.")
        model.to(device) # [Source 81]
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr) # [Source 81]

    # Log model summary
    # Need to handle device placement and remove ID from sample_batch attributes
    try:
        sample_batch.to(device) # [Source 81]
        sample_x = sample_batch.x if not isinstance(sample_batch, HeteroData) else sample_batch.x_dict # [Source 81]
        sample_edge_index = sample_batch.edge_index if not isinstance(sample_batch, HeteroData) else sample_batch.edge_index_dict # [Source 81]

        # Temporarily remove ID for summary
        if isinstance(sample_batch, HeteroData):
            id_attr_fwd = sample_batch['node', 'to', 'node'].edge_attr[:, 0].clone()
            id_attr_rev = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 0].clone()
            sample_batch['node', 'to', 'node'].edge_attr = sample_batch['node', 'to', 'node'].edge_attr[:, 1:] # [Source 81]
            sample_batch['node', 'rev_to', 'node'].edge_attr = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 1:] # [Source 81]
        else:
            id_attr = sample_batch.edge_attr[:, 0].clone()
            sample_batch.edge_attr = sample_batch.edge_attr[:, 1:] # [Source 81]

        sample_edge_attr = sample_batch.edge_attr if not isinstance(sample_batch, HeteroData) else sample_batch.edge_attr_dict # [Source 81-82]

        # Get summary (might need adjustment based on model's forward signature)
        # summary function might not work perfectly with all custom models or heterogenous data
        try:
             # For Hetero models, summary might require specific edge_type argument or adaptation
             if isinstance(sample_batch, HeteroData):
                  logging.info("Model Summary (Hetero - may be partial):")
                  # logging.info(summary(model, sample_x, sample_edge_index, sample_edge_attr)) # Often fails for hetero
                  logging.info(model) # Print model structure instead
             else:
                   logging.info("Model Summary:")
                   logging.info(summary(model, sample_x, sample_edge_index, sample_edge_attr)) # [Source 82]
        except Exception as e:
            logging.warning(f"Could not generate model summary: {e}")
            logging.info(f"Model Structure:\n{model}")


        # Add ID back to sample batch if needed elsewhere (though usually not needed after summary)
        # Restore the ID attribute if necessary (optional, depends if sample_batch is reused)
        if isinstance(sample_batch, HeteroData):
            sample_batch['node', 'to', 'node'].edge_attr = torch.cat([id_attr_fwd.unsqueeze(1), sample_batch['node', 'to', 'node'].edge_attr], dim=1)
            sample_batch['node', 'rev_to', 'node'].edge_attr = torch.cat([id_attr_rev.unsqueeze(1), sample_batch['node', 'rev_to', 'node'].edge_attr], dim=1)
        else:
            sample_batch.edge_attr = torch.cat([id_attr.unsqueeze(1), sample_batch.edge_attr], dim=1)

    except Exception as e:
        logging.error(f"Error during model summary preparation: {e}")


    # Define loss function
    if hasattr(config, 'loss') and config.loss == 'focal':
        # 使用Focal Loss
        alpha = torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device)
        gamma = config.gamma if hasattr(config, 'gamma') else 2.0
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        logging.info(f"Instantiated Focal Loss (alpha={alpha.tolist()}, gamma={gamma})")
    else:
        # 使用默认的CrossEntropyLoss
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))
        logging.info(f"Using CrossEntropyLoss with weights: {[config.w_ce1, config.w_ce2]}")

    # Start training based on model type (homo/hetero)
    if args.reverse_mp:
        model = train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config) # [Source 82]
    else:
        model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config) # [Source 82]

    # Finish wandb run if active
    if wandb.run is not None:
        logging.info(f"Finishing wandb run: {wandb.run.name}")
        wandb.finish() # [Source 82]
    else:
         logging.info("Wandb logging was disabled.")