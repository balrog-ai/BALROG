import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from balrog.agents import AgentFactory
from balrog.evaluator import EvaluatorManager
from balrog.utils import setup_environment, collect_and_summarize_results, print_summary_table, wandb_save_artifact


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    original_cwd = get_original_cwd()
    setup_environment(original_cwd=original_cwd)

    # Create an EvaluatorManager and run evaluation
    evaluator_manager = EvaluatorManager(config, original_cwd=original_cwd)
    agent_factory = AgentFactory(config)
    evaluator_manager.run(agent_factory)

    print(evaluator_manager.output_dir)
    overall_summary = collect_and_summarize_results(evaluator_manager.output_dir, config)
    print_summary_table(overall_summary)

    if config.eval.wandb_save:
        wandb_save_artifact(config)


if __name__ == "__main__":
    main()
