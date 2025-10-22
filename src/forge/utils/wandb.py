import wandb
from src.forge.utils.logger import get_logger

logger = get_logger(name="wandb")


def init_wandb(project: str, run_name: str, api:str) -> None:
    """
    Initialize wandb.

    Args:
        project (str): The name of the project.
        run_name (str): The name of the run.
        api (str): The wandb API key.
    """
    wandb.login(key=api)
    wandb.init(project=project, name=run_name)
    logger.info(f"wandb initialized: project={project}, run_name={run_name}")



