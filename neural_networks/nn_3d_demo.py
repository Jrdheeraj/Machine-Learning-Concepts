import numpy as np
import matplotlib.pyplot as plt


def relu(values):
	return np.maximum(0, values)


def build_demo_network(seed=0):
	rng = np.random.default_rng(seed)
	input_dim = 3
	hidden_dim = 5
	output_dim = 1

	weights_1 = rng.normal(size=(input_dim, hidden_dim))
	bias_1 = rng.normal(size=(hidden_dim,))
	weights_2 = rng.normal(size=(hidden_dim, output_dim))
	bias_2 = rng.normal(size=(output_dim,))

	input_vector = np.array([0.5, -1.0, 1.5])
	hidden_layer = relu(input_vector @ weights_1 + bias_1)
	output_value = hidden_layer @ weights_2 + bias_2

	return input_vector, hidden_layer, output_value


def plot_network(input_vector, hidden_layer, output_value):
	hidden_dim = hidden_layer.shape[0]

	fig = plt.figure(figsize=(9, 6))
	ax = fig.add_subplot(111, projection="3d")

	ax.scatter(
	 input_vector[0],
	 input_vector[1],
	 input_vector[2],
	 c="#16a34a",
	 s=140,
	 edgecolor="black",
	 label="Input",
	)

	hidden_x = np.linspace(-1.2, 1.2, hidden_dim)
	hidden_y = np.zeros(hidden_dim)
	hidden_z = np.full(hidden_dim, 3.0)
	hidden_positions = np.column_stack([hidden_x, hidden_y, hidden_z])

	strength = hidden_layer - hidden_layer.min()
	scale = np.ptp(hidden_layer)
	strength = strength / scale if scale > 0 else np.zeros_like(hidden_layer)

	colors = plt.cm.viridis(strength)
	ax.scatter(
	 hidden_positions[:, 0],
	 hidden_positions[:, 1],
	 hidden_positions[:, 2],
	 c=colors,
	 s=140,
	 edgecolor="black",
	 label="Hidden layer",
	)

	for index in range(hidden_dim):
		ax.plot(
		 [input_vector[0], hidden_positions[index, 0]],
		 [input_vector[1], hidden_positions[index, 1]],
		 [input_vector[2], hidden_positions[index, 2]],
		 c="#6b7280",
		 alpha=0.35,
		)

	ax.text(input_vector[0], input_vector[1], input_vector[2], "  x", color="#16a34a")
	for index in range(hidden_dim):
		ax.text(
		 hidden_positions[index, 0],
		 hidden_positions[index, 1],
		 hidden_positions[index, 2] + 0.12,
		 f"h{index + 1}",
		 fontsize=8,
		 color="#111827",
		 ha="center",
		)

	ax.scatter(
	 0.0,
	 0.0,
	 5.0,
	 c="#ef4444",
	 s=160,
	 edgecolor="black",
	 label="Output",
	)

	for position in hidden_positions:
		ax.plot(
		 [position[0], 0.0],
		 [position[1], 0.0],
		 [position[2], 5.0],
		 c="#9ca3af",
		 alpha=0.25,
		)

	ax.text(0.0, 0.0, 5.0, f"  y={float(output_value[0]):.2f}", color="#ef4444")

	ax.set_title("3D Neural Network Layer Visualization")
	ax.set_xlabel("Feature 1")
	ax.set_ylabel("Feature 2")
	ax.set_zlabel("Feature 3 / Layer Axis")
	ax.legend(loc="upper left")
	plt.tight_layout()
	plt.show()


def main():
	input_vector, hidden_layer, output_value = build_demo_network(seed=0)
	print("Input:", input_vector)
	print("Hidden activations:", np.round(hidden_layer, 3))
	print("Output:", np.round(output_value, 3))
	plot_network(input_vector, hidden_layer, output_value)


if __name__ == "__main__":
	main()
