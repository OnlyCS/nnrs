use anyhow::{ensure, Context, Result};

use crate::{edge::Edge, layer::LayerID, node::Node};

/// A neural network. Interact with this struct to create and modify your network.
#[derive(Clone)]
pub struct Network {
    pub(crate) nodes: Vec<Node>,
    pub(crate) edges: Vec<Edge>,
    pub(crate) layer_ids: Vec<LayerID>,
}

impl Network {
    pub(crate) fn get_node(&self, node_id: usize) -> Option<&Node> {
        self.nodes.iter().find(|node| node.id == node_id)
    }

    pub(crate) fn get_node_mut(&mut self, node_id: usize) -> Option<&mut Node> {
        self.nodes.iter_mut().find(|node| node.id == node_id)
    }

    pub(crate) fn get_layer(&mut self, layer_id: LayerID) -> Option<Vec<&mut Node>> {
        let matches = self
            .nodes
            .iter_mut()
            .filter(|node| node.layer_id == layer_id)
            .collect::<Vec<&mut Node>>();

        if !self.layer_ids.contains(&layer_id) {
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

    pub(crate) fn fire_layer(&mut self, id: LayerID) -> Result<()> {
        let ids = self
            .clone()
            .get_layer(id)
            .context("Layer does not exist")?
            .iter()
            .map(|node| node.id)
            .collect::<Vec<usize>>();

        let mut edges = Vec::new();

        for id in ids {
            if let Some(edge) = self
                .clone()
                .edges
                .iter()
                .find(|edge| edge.node_from_id == id)
            {
                edges.push((
                    edge.clone().node_from_id,
                    edge.clone().node_to_id,
                    edge.clone().weight,
                ))
            }
        }

        for edge in edges {
            let node_from = self.get_node(edge.0).context("Node from does not exist")?;

            let (node_from_value, node_from_threshold) = (node_from.value, node_from.threshold);

            let node_to = self
                .get_node_mut(edge.1)
                .context("Node to does not exist")?;

            if node_from_value > node_from_threshold {
                node_to.add_value(node_from_value * edge.2);
            }
        }

        Ok(())
    }

    /// Fires the network.
    pub fn fire(&mut self) -> Result<()> {
        let mut layers_sorted = self.layer_ids.clone();
        layers_sorted.sort_by(|a, b| b.cmp(a));

        for layer in layers_sorted {
            self.fire_layer(layer)?;
        }

        Ok(())
    }

    /// Creates an empty new network.
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            layer_ids: Vec::new(),
        }
    }

    /// Add a layer to the network.
    pub fn add_layer(&mut self, layer: LayerID) -> Result<()> {
        ensure!(!self.layer_ids.contains(&layer), "Layer already exists");
        self.layer_ids.push(layer);

        Ok(())
    }
}

impl Default for Network {
    /// Create a new network with an input layer and an output layer.
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            layer_ids: vec![LayerID::InputLayer, LayerID::OutputLayer],
        }
    }
}
