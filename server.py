import flwr as fl
import numpy as np

print("---------------------------------------------------------")
print("ROBUST GEN-AI SERVER: ONLINE")
print("Defense: Coordinate-wise Trimmed Mean")
print("---------------------------------------------------------")


class RobustGenAIStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_results = [
            fl.common.parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        trim_ratio    = 0.1
        num_clients   = len(weights_results)
        trim_count    = int(num_clients * trim_ratio)
        trimmed_weights = []

        for layer_idx in range(len(weights_results[0])):
            layer_weights = np.array([w[layer_idx] for w in weights_results])
            layer_weights.sort(axis=0)

            if trim_count > 0:
                trimmed_layer = layer_weights[trim_count:-trim_count]
            else:
                trimmed_layer = layer_weights[:-1]

            trimmed_weights.append(np.mean(trimmed_layer, axis=0))

        aggregated_parameters = fl.common.ndarrays_to_parameters(trimmed_weights)

        if aggregated_parameters is not None and server_round == 3:
            print("Saving robust LoRA adapter to disk...")
            np.savez("robust_adapter.npz", *trimmed_weights)

        return aggregated_parameters, {}


strategy = RobustGenAIStrategy(
    fraction_fit=1.0,
    min_fit_clients=10,
    min_available_clients=10,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
