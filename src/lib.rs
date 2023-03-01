#![warn(missing_docs)]
#![allow(clippy::ptr_arg)]

//! # nnrs
//! A simple, minimal neural network library written in Rust. No training included
//!
//! ## Example:
//! ```
//! use nnrs::{network::Network, node::Node, edge::Edge, layer::LayerID};
//!
//! let mut network = Network::default();
//! let mut output: Vec<f64> = Vec::new();
//!
//! network.add_layer(LayerID::HiddenLayer(0)).unwrap();
//!
//! let input_node_id = Node::create(&mut network, LayerID::InputLayer, 0.3).unwrap();
//! let hidden_node_id = Node::create(&mut network, LayerID::HiddenLayer(0), 0.2).unwrap();
//! let output_node_id = Node::create(&mut network, LayerID::OutputLayer, 0.0).unwrap();
//!
//! let edge_input_to_hidden = Edge::create(&mut network, input_node_id, hidden_node_id, 1.3).unwrap();
//! let edge_hidden_to_output = Edge::create(&mut network, hidden_node_id, output_node_id, 1.5).unwrap();
//!
//! network.set_inputs(vec![0.8]).unwrap();
//! network.fire().unwrap();
//! network.read(&mut output).unwrap();
//!
//! assert_eq!(output, vec![0.8 * 1.3 * 1.5]);
//! ```
//!
//! ## Neural Networks in a Nutshell
//! A neural network is a mathematical model of a biological brain.
//! It consists of nodes (neurons) and edges (synapses).
//! Nodes are grouped into layers. The input layer is the first layer, the output layer is the last layer,
//! and the layers in between are called hidden layers.
//! Nodes contain a value and a threshold. Edges have a weight. To begin, load the
//! inputs into the input layer. Then, fire the network. If the node's value is greater than the
//! threshold, the value of the node gets multiplied by the weight of the edge connecting it
//! to the next one. The resulting value is added to the next node's value. A node can have
//! multiple edges coming into it and multiple edges going out of it. When all of the nodes
//! have been fired, the output layer's values are read out.

/// Edges represent connections between nodes.
pub mod edge;

/// Contains the `LayerID` enum. Layer IDs are used to group `Nodes`.
pub mod layer;

/// Contains the `Network` struct. Use this to interact with your Network.
pub mod network;

/// Nodes are the basic building blocks of a neural network.
pub mod node;

#[test]
fn test() -> anyhow::Result<()> {
    use crate::{edge::Edge, layer::LayerID, network::Network, node::Node};

    let mut network = Network::default();
    let mut output: Vec<f64> = Vec::new();

    network.add_layer(LayerID::HiddenLayer(0))?;

    let input_node_id = Node::create(&mut network, LayerID::InputLayer, 0.3)?;
    let hidden_node_id = Node::create(&mut network, LayerID::HiddenLayer(0), 0.2)?;
    let output_node_id = Node::create(&mut network, LayerID::OutputLayer, 0.0)?;

    Edge::create(&mut network, input_node_id, hidden_node_id, 1.3)?;
    Edge::create(&mut network, hidden_node_id, output_node_id, 1.5)?;
    Edge::create(&mut network, input_node_id, output_node_id, 2.0)?;

    network.set_inputs(vec![0.8])?;
    network.fire()?;
    network.read(&mut output)?;

    assert_eq!(output, vec![(0.8 * 2.0) + (0.8 * 1.3 * 1.5)]);

    Ok(())
}
