import importlib


def configure_wandb(args, run_name_prefix, project="DyFL", entity="paridis"):
    if getattr(args, "wandb", 0) != 1:
        return None

    try:
        wandb = importlib.import_module("wandb")
    except ModuleNotFoundError:
        print("Warning: `wandb` is not installed; disabling wandb logging.")
        args.wandb = 0
        return None

    run_name = (
        f"{run_name_prefix}_{args.dataset}_num-{args.node_num}"
        f"_lepoch-{args.E}_lr-{args.lr}_note-{args.notes}"
    )
    wandb.init(project=project, name=run_name, entity=entity)
    wandb.config.update(vars(args))
    return wandb
