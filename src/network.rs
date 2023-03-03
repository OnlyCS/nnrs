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
    pub(crate) layer_ids: Vec<LayerID>,
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

        if !self.layer_ids.contains(&layer_id) {
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
    /// network.set_inputs(vec![0.8]).unwrap();
    /// ```
    pub fn set_inputs(&mut self, inputs: Vec<f64>) -> Result<()> {
        let mut input_layer = self
            .get_layer_mut(LayerID::InputLayer)
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

    /// Fires the network.
    pub fn fire(&mut self) -> Result<()> {
        let mut layers_sorted = self.layer_ids.clone();
        layers_sorted.sort();

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

    /// Adds a layer to the network.
    ///
    /// ### Example
    /// ```
    /// # use nnrs::{network::Network, layer::LayerID};
    /// # let mut network = Network::default();
    /// network.add_layer(LayerID::HiddenLayer(0));
    /// ```
    pub fn add_layer(&mut self, layer: LayerID) -> Result<()> {
        ensure!(!self.layer_ids.contains(&layer), "Layer already exists");
        self.layer_ids.push(layer);

        Ok(())
    }

    /// Reads the value of the node into the given array.
    ///
    /// ### Example
    /// ```
    /// # use nnrs::{network::Network};
    /// # let mut network = Network::default();
    /// let mut outs = Vec::new();
    /// network.read(&mut outs);
    /// ```
    pub fn read(&self, arr: &mut Vec<f64>) -> Result<()> {
        let output_layer = self
            .get_layer(LayerID::OutputLayer)
            .context("Output layer does not exist")?;
        let mut outputs = Vec::new();

        for node in output_layer {
            outputs.push(node.value);
        }

        outputs.clone_into(arr);

        Ok(())
    }

    /// Reset the network to that it can be fired again
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
    /// network.set_inputs(vec![0.8]).unwrap();
    ///
    /// let mut outs = Vec::new();
    /// let mut outs2 = Vec::new();
    ///
    /// network.read(&mut outs);
    ///
    /// network.reset();
    /// network.set_inputs(vec![0.9]).unwrap();
    /// network.read(&mut outs2);
    ///
    /// assert_eq!(outs, outs2);
    /// ```
    pub fn reset(&mut self) {
        for node in self.nodes.iter_mut() {
            node.reset();
        }
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
}

impl Default for Network {
    /// This creates a new neural network with a default configuration.
    /// (An empty input and output layer). You can also create an empty
    /// network
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            layer_ids: vec![LayerID::InputLayer, LayerID::OutputLayer],
        }
    }
}

impl Termination for Network {
    fn report(self) -> ExitCode {
        0.into()
    }
}
