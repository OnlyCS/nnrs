#![warn(missing_docs)]
#![allow(clippy::ptr_arg)]

//! # nnrs
//! A simple, minimal neural network library written in Rust.
//! Note: this library is still in development, and is not yet ready for use.
//!
//! ## Example:
//! ```
//! use nnrs::{network::Network, node::Node, edge::Edge, layer::LayerID, activationfn::ActivationFn};
//!
//! let mut network = Network::create(1, 1, ActivationFn::Linear).unwrap();
//! let mut output: Vec<f64> = Vec::new();
//!
//! let layer_id = network.add_layer();
//!
//! let input_node_id = network.input_node_ids().pop().unwrap();
//! let hidden_node_id = Node::create(&mut network, layer_id, 0.2, 0.0).unwrap();
//! let output_node_id = network.output_node_ids().pop().unwrap();
//!
//! let edge_input_to_hidden = Edge::create(&mut network, input_node_id, hidden_node_id, 1.3).unwrap();
//! let edge_hidden_to_output = Edge::create(&mut network, hidden_node_id, output_node_id, 1.5).unwrap();
//!
//! network.fire(vec![0.8], &mut output).unwrap();
//!
//! assert_eq!(output, vec![0.8 * 1.3 * 1.5]);
//! ```

/// Edges represent connections between nodes.
pub mod edge;

/// Contains the `LayerID` enum. Layer IDs are used to group `Nodes`.
pub mod layer;

/// Contains the `Network` struct. Use this to interact with your Network.
pub mod network;

/// Nodes are the basic building blocks of a neural network.
pub mod node;

/// NEAT training for the Neural Network
#[cfg(feature = "neat")]
pub mod neat;

/// Activation functions.
pub mod activationfn;

#[test]
fn test_creation() -> anyhow::Result<crate::network::Network> {
    use crate::{activationfn::ActivationFn, edge::Edge, network::Network, node::Node};

    let mut network = Network::create(1, 1, ActivationFn::ReLU)?;
    let hidden_id = network.add_layer();

    let input_node_id = network.input_node_ids().pop().unwrap();
    let hidden_node_id = Node::create(&mut network, hidden_id, 0.2, 0.0)?;
    let output_node_id = network.output_node_ids().pop().unwrap();

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
//         activationfn::ActivationFn,
//         neat::{
//             environment::Environment,
//             settings::{Settings, TrainingMode},
//         },
//         network::Network,
//     };

//     let network = Network::create(2, 1, ActivationFn::Linear)?;

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
