#![warn(missing_docs)]

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
//! let input_node_id = Node::create(&mut network, LayerID::InputLayer, 0.0).unwrap();
//! let hidden_node_id = Node::create(&mut network, LayerID::HiddenLayer, 0.0).unwrap();
//! let edge_id = Edge::create(&mut network, input_node_id, hidden_node_id, 0.0).unwrap();
//!
//! network.set_node_value(input_node_id, 1.0).unwrap();
//!
//! // todo:
//! // network.fire();
//! // network.read(&mut output);
//! ```

extern crate self as nnrs;

/// A neural network edge.
pub mod edge;

/// Contains the `LayerID` enum. Layer IDs are used to group `Nodes`.
pub mod layer;

/// Contains the `Network` struct. Use this to interact with your Network.
pub mod network;

/// A neural network node.
pub mod node;
