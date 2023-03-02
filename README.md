# NNRS - Rust Neural Network Library
NNRS is a Rust library for creating and working with feedforward neural
networks. It provides a set of tools for building and manipulating neural
networks, including creating nodes and edges, setting inputs, and firing the
network to generate outputs.

## Installation
To use NNRS, simply add it as a dependency to your Rust project by adding the
following line to your `Cargo.toml` file:

```toml
[dependencies]
nnrs = "0.1.0"
```

## Usage

### Creating a Network
To create a new neural network, use the `Network` struct:

```rust
use nnrs::{network::Network, node::Node, edge::Edge, layer::LayerID};

let mut network = Network::default();
```

This creates a new neural network with a default configuration. 
(An empty input and output layer). You can also create an empty
network

```rust
let mut network = Network::empty();
```

### Creating Layers
Layers are used to group nodes together. To create a layer, use the `Layer` struct:

```rust
network.add_layer(LayerID::InputLayer);
network.add_layer(LayerID::HiddenLayer(0));
network.add_layer(LayerID::OutputLayer);
```

### Creating Nodes
Nodes are the basic building blocks of a neural network. To create a node, use the `Node::create` method:

```rust
let input_node_id = Node::create(&mut network, LayerID::InputLayer, 0.3)?;
let hidden_node_id = Node::create(&mut network, LayerID::HiddenLayer(0), 0.2)?;
let output_node_id = Node::create(&mut network, LayerID::OutputLayer, 0.0)?;
```

This creates two nodes, one in the input layer and one in the output layer. The first argument is the network. The second argument is the layer the node should be created in. The third argument is the threshold of the node. An `OutputNode`'s
threshold is ignored.

### Creating Edges
Edges represent the connections between nodes in a neural network. To create an edge, use the `Edge::create` method:

```rust
Edge::create(&mut network, input_node_id, hidden_node_id, 1.3)?;
Edge::create(&mut network, hidden_node_id, output_node_id, 1.5)?;
Edge::create(&mut network, input_node_id, output_node_id, 2.0)?;
```

This creates three edges, one going from the input to the output node and a
weight of 2.0, and two going from the input to the hidden node and the hidden
node to the output node, with weights of 1.3 and 1.5 respectively.

### Setting Inputs
To set the inputs of a neural network, use the `Network::set_inputs` method:

```rust
network.set_inputs(&[0.8])?;
```

This sets the input of the input node to 0.8.

### Firing the Network
To fire the network, use the `Network::fire` method:

```rust
network.fire()?;
```

### Getting Outputs

To get the outputs of a neural network, use the `Network::get_outputs` method:

```rust
let mut output: Vec<f64> = Vec::new();

network.read(&mut output)?;
```

This gets the output of the output node and stores it in the `output` vector.

### Serializing and Deserializing

To serialize a network, use the `Network::serialize` method:

```rust
let serialized: String = network.serialize()?;
```

To deserialize a network, use the `Network::deserialize` method:

```rust
let mut network: Network = Network::deserialized(&serialized)?;
```


## Limitations

At this moment, NNRS does not include training functionality. You
can use this library to generate outputs from pre-trained networks.

## Roadmap

- [x] Basic Parts
- [x] Generating Outputs
- [ ] Serialization
- [ ] Training (NEAT?)

## License
NNRS is licensed under the AGPLv3 license. See the `LICENSE` file for more
information.
