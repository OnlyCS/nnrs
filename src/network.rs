use anyhow::{ensure, Context, Result};

use crate::{edge::Edge, layer::LayerID, node::Node};

/// A neural network. Interact with this struct to create and modify your network.
#[derive(Clone)]
pub struct Network {
    pub(crate) nodes: Vec<Node>,
    pub(crate) edges: Vec<Edge>,
}

impl Network {
    pub(crate) fn get_node(&self, node_id: usize) -> Option<&Node> {
        self.nodes.iter().find(|node| node.id == node_id)
    }

    pub(crate) fn get_layer(&mut self, layer_id: LayerID) -> Option<Vec<&mut Node>> {
        let matches = self
            .nodes
            .iter_mut()
            .filter(|node| node.layer_id == layer_id)
            .collect::<Vec<&mut Node>>();

        if matches.is_empty() {
            None
        } else {
            Some(matches)
        }
    }

    pub(crate) fn get_edge(&self, edge_id: usize) -> Option<&Edge> {
        self.edges.iter().find(|edge| edge.id == edge_id)
    }

    /// Sets the inputs of the network.
    pub fn set_inputs(&mut self, inputs: Vec<f64>) -> Result<()> {
        let mut input_layer = self
            .get_layer(LayerID::InputLayer)
            .context("Input layer does not exist")?;

        ensure!(
            inputs.len() == input_layer.len(),
            "Number of inputs must match number of input nodes"
        );

        for (input, node) in inputs.iter().zip(input_layer.iter_mut()) {
            node.set_value(*input)?;
        }

        Ok(())
    }
}
