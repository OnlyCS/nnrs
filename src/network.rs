use std::{
    fs::File,
    io::{Read, Write},
    process::{ExitCode, Termination},
};

use crate::{activationfn::ActivationFn, edge::Edge, layer::LayerID, node::Node};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

/// A neural network. Interact with this struct to create and modify your network.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Network {
    pub(crate) nodes: Vec<Node>,
    pub(crate) edges: Vec<Edge>,
    pub(crate) layers: Vec<LayerID>,
    pub(crate) fitness: Option<f64>,
    pub(crate) activation_fn: ActivationFn,
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
    /// # use nnrs::{network::Network, node::Node, layer::LayerID, edge::Edge, activationfn::ActivationFn};
    /// # let mut network = Network::create(1, 1, ActivationFn::Sigmoid).unwrap();
    /// # let layerid = network.add_layer();
    /// # let input_node_id = network.input_node_ids().pop().unwrap();
    /// # let hidden_node_id = Node::create(&mut network, layerid, 0.0).unwrap();
    /// # let output_node_id = network.output_node_ids().pop().unwrap();
    /// # Edge::create(&mut network, input_node_id, hidden_node_id, 0.5).unwrap();
    /// # Edge::create(&mut network, hidden_node_id, output_node_id, 0.5).unwrap();
    /// let mut output = vec![];
    /// network.fire(vec![0.8], &mut output).unwrap();
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

        for (node_from_id, node_to_id, edge_weight) in edges {
            let node_from_value = self
                .get_node(node_from_id)
                .context("Node from does not exist")?
                .value;

            let node_to = self
                .get_node_mut(node_to_id)
                .context("Node to does not exist")?;

            node_to.add_value(node_from_value * edge_weight);
        }

        // get the next layer's id
        let mut next_layer = match id {
            LayerID::InputLayer => LayerID::HiddenLayer(0),
            LayerID::HiddenLayer(id) => LayerID::HiddenLayer(id + 1),
            LayerID::OutputLayer => LayerID::OutputLayer,
        };

        if !self.layers.contains(&next_layer) {
            next_layer = LayerID::OutputLayer;
        }

        let mut layer = self
            .get_layer_mut(next_layer)
            .context("Layer does not exist")?;

        let nodes = layer
            .iter_mut()
            .filter(|node| node.layer_id == next_layer)
            .collect::<Vec<&mut &mut Node>>();

        for node in nodes {
            node.add_value(node.bias);
            node.value = node.activation_fn.run(node.value);
        }

        Ok(())
    }

    /// Adds the next available layer.
    ///
    /// ### Example
    /// ```
    /// # use nnrs::{network::Network, layer::LayerID, activationfn::ActivationFn};
    /// # let mut network = Network::create(1, 1, ActivationFn::Sigmoid).unwrap();
    /// let layer_id: LayerID = network.add_layer();
    /// ```
    pub fn add_layer(&mut self) -> LayerID {
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

        self.layers.push(next_layer);

        next_layer
    }

    /// Serialize the network to a string
    ///
    /// ### Example
    /// ```
    /// # use nnrs::{network::Network, node::Node, layer::LayerID, edge::Edge, activationfn::ActivationFn};
    /// # let mut network = Network::create(1, 1, ActivationFn::Sigmoid).unwrap();
    /// # let layerid = network.add_layer();
    /// # let input_node_id = network.input_node_ids().pop().unwrap();
    /// # let hidden_node_id = Node::create(&mut network, layerid, 0.0).unwrap();
    /// # let output_node_id = network.output_node_ids().pop().unwrap();
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
    /// # use nnrs::{network::Network, node::Node, layer::LayerID, edge::Edge, activationfn::ActivationFn};
    /// # let mut network = Network::create(1, 1, ActivationFn::Sigmoid).unwrap();
    /// # let layer_id = network.add_layer();
    /// # let input_node_id = network.input_node_ids().pop().unwrap();
    /// # let hidden_node_id = Node::create(&mut network, LayerID::HiddenLayer(0), 0.0).unwrap();
    /// # let output_node_id = network.output_node_ids().pop().unwrap();
    /// # Edge::create(&mut network, input_node_id, hidden_node_id, 0.5).unwrap();
    /// # Edge::create(&mut network, hidden_node_id, output_node_id, 0.5).unwrap();
    /// let string = network.serialize().unwrap();
    /// let mut network2 = Network::deserialized(&string).unwrap();
    ///
    /// let mut outs = Vec::new();
    /// let mut outs2 = Vec::new();
    ///
    /// network.fire(vec![0.8], &mut outs).unwrap();
    /// network2.fire(vec![0.8], &mut outs2).unwrap();
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
    /// # use nnrs::{network::Network, node::Node, layer::LayerID, edge::Edge, activationfn::ActivationFn};
    /// # let mut network = Network::create(1, 1, ActivationFn::Sigmoid).unwrap();
    /// # let layerid = network.add_layer();
    /// # let input_node_id = network.input_node_ids().pop().unwrap();
    /// # let hidden_node_id = Node::create(&mut network, layerid, 0.0).unwrap();
    /// # let output_node_id = network.output_node_ids().pop().unwrap();
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
    /// # use nnrs::{network::Network, node::Node, layer::LayerID, edge::Edge, activationfn::ActivationFn};
    /// # let mut network = Network::create(1, 1, ActivationFn::Sigmoid).unwrap();
    /// # let layerid = network.add_layer();
    /// # let input_node_id = network.input_node_ids().pop().unwrap();
    /// # let hidden_node_id = Node::create(&mut network, layerid, 0.0).unwrap();
    /// # let output_node_id = network.output_node_ids().pop().unwrap();
    /// # Edge::create(&mut network, input_node_id, hidden_node_id, 0.5).unwrap();
    /// # Edge::create(&mut network, hidden_node_id, output_node_id, 0.5).unwrap();
    /// network.save("network.json").unwrap();
    /// let mut network2 = Network::load("network.json").unwrap();
    ///
    /// let mut outs = Vec::new();
    /// let mut outs2 = Vec::new();
    ///
    /// network.fire(vec![0.8], &mut outs).unwrap();
    /// network2.fire(vec![0.8], &mut outs2).unwrap();
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
    /// # use nnrs::{network::Network, activationfn::ActivationFn};
    /// let mut network = Network::create(2, 1, ActivationFn::Linear).unwrap();
    /// ```
    pub fn create(input_ct: usize, output_ct: usize, activation_fn: ActivationFn) -> Result<Self> {
        let mut network = Self {
            edges: vec![],
            nodes: vec![],
            layers: vec![LayerID::InputLayer, LayerID::OutputLayer],
            fitness: None,
            activation_fn,
        };

        let mut input_ids = Vec::new();
        let mut output_ids = Vec::new();

        for _ in 0..input_ct {
            input_ids.push(Node::create(&mut network, LayerID::InputLayer, 0.0)?);
        }

        for _ in 0..output_ct {
            output_ids.push(Node::create(&mut network, LayerID::OutputLayer, 0.0)?);
        }

        Ok(network)
    }

    /// Get the ids of all the input nodes
    ///
    /// ### Example
    /// ```
    /// # use nnrs::{network::Network, node::Node, layer::LayerID, edge::Edge, activationfn::ActivationFn};
    /// let mut network = Network::create(1, 1, ActivationFn::Linear).unwrap();
    /// let input_node_id = network.input_node_ids().pop().unwrap();
    pub fn input_node_ids(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.layer_id == LayerID::InputLayer)
            .map(|n| n.id)
            .collect()
    }

    /// Get the ids of all the output nodes
    ///
    /// ### Example
    /// ```
    /// # use nnrs::{network::Network, node::Node, layer::LayerID, edge::Edge, activationfn::ActivationFn};
    /// let mut network = Network::create(1, 1, ActivationFn::Linear).unwrap();
    /// let output_node_id = network.output_node_ids().pop().unwrap();
    /// ```
    pub fn output_node_ids(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.layer_id == LayerID::OutputLayer)
            .map(|n| n.id)
            .collect()
    }
}

impl Termination for Network {
    fn report(self) -> ExitCode {
        0.into()
    }
}
