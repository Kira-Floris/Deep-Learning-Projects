import pickle
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl

from utils.dataset import prepare_dataset
from utils.client import generate_client_function
from utils.server import get_on_fit_config, get_evaluate_fn

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    save_path = HydraConfig.get().runtime.output_dir

    train_generators, val_generators, test_generator, vocab_size, max_sequence_len = prepare_dataset(cfg.num_clients, cfg.batch_size, cfg.val_ratio)

    client_fn = generate_client_function(train_generators, val_generators, vocab_size, cfg.embedding_dim, cfg.rnn_units)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.0,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ),
        evaluate_fn=get_evaluate_fn(vocab_size, test_generator)
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),
        strategy=strategy
    )
    results_path = Path(save_path) / "results.pkl"
    results = {"history": history, "anythingelse":"here"}
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()