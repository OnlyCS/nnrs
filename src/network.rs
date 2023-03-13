use std::{
    fs::File,
    io::{Read, Write},
    process::{ExitCode, Termination},
};

use crate::{edge::Edge, layer::LayerID, node::Node};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

/// A neural network. Interact with this struct to create and modify your network.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Network {
    pub(crate) nodes: Vec<Node>,
    pub(crate) edges: Vec<Edge>,
    pub(crate) layers: Vec<LayerID>,
    pub(crate) fitness: Option<f64>,
}

impl Network {
    pub(crate) fn get_node(&self, node_id: usize) -> Option<&Node> {
        self.nodes.iter().find(|node| node.id == node_id)
    }

    pub(crate) fn get_node_mut(&mut self, node_id: usize) -> Option<&mut Node> {
        self.nodes.iter_mut().find(|node| node.id == node_id)
    }

    pub(crate) fn get_layer(&self, layer_id: LayerID) -> Option<Vec<&Node>> {
        let matches = self
            .nodes
            .iter()
            .filter(|node| node.layer_id == layer_id)
            .collect::<Vec<&Node>>();

        if !self.layers.contains(&layer_id) {
            None
        } else {
            Some(matches)
        }
    }

    pub(crate) fn get_layer_mut(&mut self, layer_id: LayerID) -> Option<Vec<&mut Node>> {
        let matches = self
            .nodes
            .iter_mut()
            .filter(|node| node.layer_id == layer_id)
            .collect::<Vec<&mut Node>>();

        if !self.layers.contains(&layer_id) {
            None
        } else {
            Some(matches)
        }
    }

    pub(crate) fn get_edge(&self, edge_id: usize) -> Option<&Edge> {
        self.edges.iter().find(|edge| edge.id == edge_id)
    }

    /// Runs the inputs of the network.
    ///
    /// ### Example
    /// ```
    /// # use nnrs::{network::Network, node::Node, layer::LayerID, edge::Edge};
    /// # let mut network = Network::default();
    /// # network.add_layer(LayerID::HiddenLayer(0)).unwrap();
    /// # let input_node_id = Node::create(&mut network, LayerID::InputLayer, 0.0).unwrap();
    /// # let hidden_node_id = Node::create(&mut network, LayerID::HiddenLayer(0), 0.0).unwrap();
    /// # let output_node_id = Node::create(&mut network, LayerID::OutputLayer, 0.0).unwrap();
    /// # Edge::create(&mut network, input_node_id, hidden_node_id, 0.5).unwrap();
    /// # Edge::create(&mut network, hidden_node_id, output_node_id, 0.5).unwrap();
    /// let mut output = vec![];
    /// network.set_inputs(vec![0.8], &mut output).unwrap();
    /// ```
    pub fn fire(&mut self, inputs: Vec<f64>, outputs: &mut Vec<f64>) -> Result<()> {
        ensure!(
            self.nodes
                .iter()
                .filter(|node| node.layer_id == LayerID::InputLayer)
                .count()
                == inputs.len(),
            "Number of inputs does not match number of input nodes"
        );

        for (node, input) in self
            .nodes
            .iter_mut()
            .filter(|node| node.layer_id == LayerID::InputLayer)
            .zip(inputs)
        {
            node.value = input;
        }

        for layer_id in self.layers.clone().iter() {
            self.fire_layer(*layer_id)?;
        }

        let output_layer = self
            .get_layer(LayerID::OutputLayer)
            .context("Output layer does not exist")?
            .iter()
            .map(|node| node.value)
            .collect::<Vec<f64>>();

        outputs.clear();
        outputs.extend(output_layer);

        // clear all node's `value` fields
        for node in self.nodes.iter_mut() {
            node.reset();
        }

        Ok(())
    }

    pub(crate) fn fire_layer(&mut self, id: LayerID) -> Result<()> {
        let ids = self
            .clone()
            .get_layer_mut(id)
            .context("Layer does not exist")?
            .iter()
            .map(|node| node.id)
            .collect::<Vec<usize>>();

        let mut edges = Vec::new();

        for id in ids {
            for edge in self
                .clone()
                .edges
                .iter()
                .filter(|edge| edge.node_from_id == id)
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

    /// Adds the next available layer.
    ///
    /// ### Example
    /// ```
    /// # use nnrs::{network::Network, layer::LayerID};
    /// # let mut network = Network::create();
    /// let layer_id: LayerID = network.add_layer();
    /// ```
    pub fn add_layer(&mut self) -> Result<LayerID> {
        let mut hidden_layers = self
            .layers
            .iter()
            .filter(|layer| matches!(layer, LayerID::HiddenLayer(_)))
            .collect::<Vec<&LayerID>>();

        hidden_layers.sort();

        let next_layer = match hidden_layers.last() {
            Some(layer) => match layer {
                LayerID::HiddenLayer(id) => LayerID::HiddenLayer(id + 1),
                _ => unreachable!(),
            },
            None => LayerID::HiddenLayer(0),
        };

        ensure!(!self.layers.contains(&next_layer), "Layer already exists");
        self.layers.push(next_layer);

        Ok(next_layer)
    }

    /// Serialize the network to a string
    ///
    /// ### Example
    /// ```
    /// # use nnrs::{network::Network, node::Node, layer::LayerID, edge::Edge};
    /// # let mut network = Network::default();
    /// # network.add_layer(LayerID::HiddenLayer(0)).unwrap();
    /// # let input_node_id = Node::create(&mut network, LayerID::InputLayer, 0.0).unwrap();
    /// # let hidden_node_id = Node::create(&mut network, LayerID::HiddenLayer(0), 0.0).unwrap();
    /// # let output_node_id = Node::create(&mut network, LayerID::OutputLayer, 0.0).unwrap();
    /// # Edge::create(&mut network, input_node_id, hidden_node_id, 0.5).unwrap();
    /// # Edge::create(&mut network, hidden_node_id, output_node_id, 0.5).unwrap();
    /// let string = network.serialize().unwrap();
    /// ```
    pub fn serialize(&self) -> Result<String> {
        serde_json::to_string(&self).context("Could not serialize network")
    }

    /// Deserialize a network from a string
    ///
    /// ### Example
    /// ```
    /// # use nnrs::{network::Network, node::Node, layer::LayerID, edge::Edge};
    /// # let mut network = Network::default();
    /// # network.add_layer(LayerID::HiddenLayer(0)).unwrap();
    /// # let input_node_id = Node::create(&mut network, LayerID::InputLayer, 0.0).unwrap();
    /// # let hidden_node_id = Node::create(&mut network, LayerID::HiddenLayer(0), 0.0).unwrap();
    /// # let output_node_id = Node::create(&mut network, LayerID::OutputLayer, 0.0).unwrap();
    /// # Edge::create(&mut network, input_node_id, hidden_node_id, 0.5).unwrap();
    /// # Edge::create(&mut network, hidden_node_id, output_node_id, 0.5).unwrap();
    /// # let string = network.serialize().unwrap();
    /// let mut network2 = Network::deserialized(&string).unwrap();
    ///
    /// let mut outs = Vec::new();
    /// let mut outs2 = Vec::new();
    ///
    /// network.set_inputs(vec![0.8]).unwrap();
    /// network2.set_inputs(vec![0.8]).unwrap();
    ///
    /// network.read(&mut outs);
    /// network2.read(&mut outs2);
    ///
    /// assert_eq!(outs, outs2);
    /// ```
    pub fn deserialized(string: &str) -> Result<Self> {
        serde_json::from_str(string).context("Could not deserialize network")
    }

    /// Serialize the network to a file
    ///
    /// ### Example
    /// ```
    /// # use nnrs::{network::Network, node::Node, layer::LayerID, edge::Edge};
    /// # let mut network = Network::default();
    /// # network.add_layer(LayerID::HiddenLayer(0)).unwrap();
    /// # let input_node_id = Node::create(&mut network, LayerID::InputLayer, 0.0).unwrap();
    /// # let hidden_node_id = Node::create(&mut network, LayerID::HiddenLayer(0), 0.0).unwrap();
    /// # let output_node_id = Node::create(&mut network, LayerID::OutputLayer, 0.0).unwrap();
    /// # Edge::create(&mut network, input_node_id, hidden_node_id, 0.5).unwrap();
    /// # Edge::create(&mut network, hidden_node_id, output_node_id, 0.5).unwrap();
    /// network.save("network.json").unwrap();
    /// ```
    pub fn save(&self, path: &str) -> Result<()> {
        let mut file = File::create(path).context("Could not create file")?;
        file.write_all(self.serialize()?.as_bytes())
            .context("Could not write to file")?;
        Ok(())
    }

    /// Deserialize a network from a file
    ///
    /// ### Example
    /// ```
    /// # use nnrs::{network::Network, node::Node, layer::LayerID, edge::Edge};
    /// # let mut network = Network::default();
    /// # network.add_layer(LayerID::HiddenLayer(0)).unwrap();
    /// # let input_node_id = Node::create(&mut network, LayerID::InputLayer, 0.0).unwrap();
    /// # let hidden_node_id = Node::create(&mut network, LayerID::HiddenLayer(0), 0.0).unwrap();
    /// # let output_node_id = Node::create(&mut network, LayerID::OutputLayer, 0.0).unwrap();
    /// # Edge::create(&mut network, input_node_id, hidden_node_id, 0.5).unwrap();
    /// # Edge::create(&mut network, hidden_node_id, output_node_id, 0.5).unwrap();
    /// # network.save("network.json").unwrap();
    /// let mut network2 = Network::load("network.json").unwrap();
    ///
    /// let mut outs = Vec::new();
    /// let mut outs2 = Vec::new();
    ///
    /// network.set_inputs(vec![0.8]).unwrap();
    /// network.set_inputs(vec![0.8]).unwrap();
    ///
    /// network.read(&mut outs);
    /// network2.read(&mut outs2);
    ///
    /// assert_eq!(outs, outs2);
    /// ```
    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path).context("Could not open file")?;
        let mut string = String::new();
        file.read_to_string(&mut string)
            .context("Could not read file")?;
        Self::deserialized(&string)
    }

    /// Create a new network with the given number of inputs and outputs
    ///
    /// ### Example
    /// ```
    /// # use nnrs::network::Network;
    /// let network = Network::create(2, 1).unwrap();
    /// ```
    pub fn create(input_ct: usize, output_ct: usize) -> Result<Self> {
        let mut network = Self {
            edges: vec![],
            nodes: vec![],
            layers: vec![LayerID::InputLayer, LayerID::OutputLayer],
            fitness: None,
        };

        for _ in 0..input_ct {
            Node::create(&mut network, LayerID::InputLayer, 0.0)?;
        }

        for _ in 0..output_ct {
            Node::create(&mut network, LayerID::OutputLayer, 0.0)?;
        }

        Ok(network)
    }
}

impl Termination for Network {
    fn report(self) -> ExitCode {
        0.into()
    }
}
