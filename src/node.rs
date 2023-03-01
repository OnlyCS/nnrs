use anyhow::{ensure, Result};

use crate::{layer::LayerID, network::Network};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum NodeType {
    InputNode,
    HiddenNode,
    OutputNode,
}

#[derive(Clone)]
pub struct Node {
    pub(crate) node_type: NodeType,
    pub(crate) id: usize,
    pub(crate) layer_id: LayerID,
    pub(crate) value: f64,
    pub(crate) threshold: f64,
}

impl Node {
    pub fn create(
        network: &mut Network,
        id: usize,
        layer_id: LayerID,
        threshold: f64,
    ) -> Result<()> {
        ensure!(
            network.get_node(id).is_none(),
            "Node with id {} already exists",
            id
        );
        ensure!(
            network.get_layer(layer_id).is_some(),
            "Layer with id {:?} does not exist",
            layer_id
        );

        let node_type = match layer_id {
            LayerID::InputNode => NodeType::InputNode,
            LayerID::OutputNode => NodeType::OutputNode,
            LayerID::HiddenNode(_) => NodeType::HiddenNode,
        };

        let node = Node {
            node_type,
            id,
            layer_id,
            value: 0.0,
            threshold,
        };

        network.nodes.push(node);

        Ok(())
    }

    pub fn set_value(&mut self, value: f64) -> Result<()> {
        ensure!(
            self.node_type == NodeType::InputNode,
            "Cannot set value of non-input node"
        );

        self.value = value;

        Ok(())
    }

    pub fn add_value(&mut self, value: f64) {
        self.value += value;
    }

    pub fn get_id(&self) -> usize {
        self.id
    }

    pub fn get_layer_id(&self) -> LayerID {
        self.layer_id
    }

    pub fn get_value(&self) -> f64 {
        self.value
    }

    pub fn get_threshold(&self) -> f64 {
        self.threshold
    }

    pub fn get_node_type(&self) -> NodeType {
        self.node_type
    }
}
