use anyhow::{ensure, Result};

use crate::{layer::LayerID, network::Network};

/// Possible node types.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum NodeType {
    /// An input node.
    InputNode,

    /// A hidden (middle) node.
    HiddenNode,

    /// An output node.
    OutputNode,
}

/// A neural network node.
#[derive(Clone)]
pub struct Node {
    pub(crate) node_type: NodeType,
    pub(crate) id: usize,
    pub(crate) layer_id: LayerID,
    pub(crate) value: f64,
    pub(crate) threshold: f64,
}

impl Node {
    /// Creates a new node.
    pub fn create(network: &mut Network, layer_id: LayerID, threshold: f64) -> Result<usize> {
        let id = network.nodes.len();

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
            threshold,
        };

        let id = node.id;

        network.nodes.push(node);

        Ok(id)
    }

    pub(crate) fn set_value(&mut self, value: f64) -> Result<()> {
        ensure!(
            self.node_type == NodeType::InputNode,
            "Cannot set value of non-input node"
        );

        self.value = value;

        Ok(())
    }

    pub(crate) fn add_value(&mut self, value: f64) {
        self.value += value;
    }
}
