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

// /// NEAT training for the Neural Network
// #[cfg(feature = "neat")]
// pub mod neat;

#[test]
fn test_creation() -> anyhow::Result<crate::network::Network> {
    use crate::{edge::Edge, layer::LayerID, network::Network, node::Node};

    let mut network = Network::create(1, 1)?;
    let hidden_id = network.add_layer()?;

    let input_node_id = Node::create(&mut network, LayerID::InputLayer, 0.3)?;
    let hidden_node_id = Node::create(&mut network, hidden_id, 0.2)?;
    let output_node_id = Node::create(&mut network, LayerID::OutputLayer, 0.0)?;

    Edge::create(&mut network, input_node_id, hidden_node_id, 1.3)?;
    Edge::create(&mut network, hidden_node_id, output_node_id, 1.5)?;
    Edge::create(&mut network, input_node_id, output_node_id, 2.0)?;

    Ok(network)
}

#[test]
fn test_io() -> anyhow::Result<()> {
    let mut network = test_creation()?;
    let mut output: Vec<f64> = Vec::new();

    network.fire(vec![0.8], &mut output)?;

    assert_eq!(output, vec![(0.8 * 2.0) + (0.8 * 1.3 * 1.5)]);

    Ok(())
}

#[test]
fn test_serialization() -> anyhow::Result<()> {
    use crate::network::Network;

    let mut network = test_creation()?;

    let serialized = serde_json::to_string(&network)?;
    let mut deserialized: Network = serde_json::from_str(&serialized)?;

    let mut expected_out = Vec::new();
    let mut actual_out = Vec::new();

    network.fire(vec![0.8], &mut expected_out)?;
    deserialized.fire(vec![0.8], &mut actual_out)?;

    assert_eq!(expected_out, actual_out);

    Ok(())
}

// #[test]
// fn test_neat() -> anyhow::Result<()> {
//     use crate::{
//         neat::{
//             environment::Environment,
//             settings::{Settings, TrainingMode},
//         },
//         network::Network,
//     };

//     let network = Network::create(2, 1)?;

//     let settings = Settings {
//         population_size: 100,
//         training_mode: TrainingMode::FitnessTarget(100f64),
//         ..Settings::default()
//     };

//     let mut environment = Environment::new(settings, network, |network| {
//         let mut distance = 0.0;
//         let mut output = vec![];

//         network.fire(vec![0.0, 0.0], &mut output).unwrap();
//         distance += (0f64 - output[0]).abs();
//         output.clear();

//         network.fire(vec![0.0, 1.0], &mut output).unwrap();
//         distance += (1f64 - output[0]).abs();
//         output.clear();

//         network.fire(vec![1.0, 0.0], &mut output).unwrap();
//         distance += (1f64 - output[0]).abs();
//         output.clear();

//         network.fire(vec![1.0, 1.0], &mut output).unwrap();
//         distance += (0f64 - output[0]).abs();
//         output.clear();

//         (4f64 - distance).powi(2)
//     });

//     let mut champ = environment.run().unwrap();
//     let mut output = vec![];

//     champ.fire(vec![0.0, 0.0], &mut output).unwrap();

//     println!("Campion says xor of 1 and 0 is: {:?}", output);

//     Ok(())
// }
