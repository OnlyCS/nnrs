use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use crate::{activationfn::ActivationFn, layer::LayerID, network::Network};

/// Possible node types.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum NodeType {
    /// An input node.
    InputNode,

    /// A hidden (middle) node.
    HiddenNode,

    /// An output node.
    OutputNode,
}

/// Nodes are the basic building blocks of a neural network.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Node {
    pub(crate) node_type: NodeType,
    pub(crate) id: usize,
    pub(crate) layer_id: LayerID,
    pub(crate) value: f64,
    pub(crate) bias: f64,
    pub(crate) activation_fn: ActivationFn,
}

impl Node {
    /// Creates a new node.
    ///
    /// ### Examples
    /// ```
    /// # use nnrs::{network::Network, node::Node, edge::Edge, layer::LayerID, activationfn::ActivationFn};
    /// # let mut network = Network::create(1, 1, ActivationFn::Sigmoid).unwrap();
    /// # let layerid = network.add_layer();
    /// let input_node_id = network.input_node_ids().pop().unwrap();
    /// let hidden_node_id = Node::create(&mut network, layerid, 0.0).unwrap();
    /// let output_node_id = network.output_node_ids().pop().unwrap();
    /// ```
    pub fn create(network: &mut Network, layer_id: LayerID, bias: f64) -> Result<usize> {
        Self::create_with_custom_activation(network, layer_id, bias, network.activation_fn)
    }

    /// Creates a new node with a custom activation function.
    ///
    /// ### Examples
    /// ```
    /// # use nnrs::{network::Network, node::Node, edge::Edge, layer::LayerID, activationfn::ActivationFn};
    /// # let mut network = Network::create(1, 1, ActivationFn::Sigmoid).unwrap();
    /// # let layerid = network.add_layer();
    /// let input_node_id = Node::create_with_custom_activation(&mut network, layerid, 0.0, ActivationFn::Sigmoid).unwrap();
    pub fn create_with_custom_activation(
        network: &mut Network,
        layer_id: LayerID,
        bias: f64,
        activation_fn: ActivationFn,
    ) -> Result<usize> {
        let id = network.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;

        ensure!(
            network.get_node(id).is_none(),
            "Node with id {} already exists",
            id
        );
        ensure!(
            network.get_layer_mut(layer_id).is_some(),
            "Layer with id {:?} does not exist",
            layer_id
        );

        let node_type = match layer_id {
            LayerID::InputLayer => NodeType::InputNode,
            LayerID::OutputLayer => NodeType::OutputNode,
            LayerID::HiddenLayer(_) => NodeType::HiddenNode,
        };

        let node = Node {
            node_type,
            id,
            layer_id,
            value: 0.0,
            bias,
            activation_fn,
        };

        let id = node.id;

        network.nodes.push(node);

        Ok(id)
    }

    pub(crate) fn add_value(&mut self, value: f64) {
        self.value += value;
    }

    pub(crate) fn reset(&mut self) {
        self.value = 0.0;
    }
}
