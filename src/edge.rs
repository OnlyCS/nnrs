use crate::network::Network;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

/// Edges represent connections between nodes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub(crate) id: usize,
    pub(crate) weight: f64,
    pub(crate) node_from_id: usize,
    pub(crate) node_to_id: usize,
}

impl Edge {
    /// Creates a new edge.
    ///
    /// ### Examples
    /// ```
    /// # use nnrs::{network::Network, node::Node, edge::Edge, layer::LayerID, activationfn::ActivationFn};
    /// # let mut network = Network::create(1, 1, ActivationFn::Sigmoid).unwrap();
    /// # let layerid = network.add_layer();
    /// # let input_node_id = network.input_node_ids().pop().unwrap();
    /// # let hidden_node_id = Node::create(&mut network, layerid, 0.0).unwrap();
    /// # let output_node_id = network.output_node_ids().pop().unwrap();
    /// Edge::create(&mut network, input_node_id, hidden_node_id, 1.3);
    /// Edge::create(&mut network, hidden_node_id, output_node_id, 1.5);
    /// Edge::create(&mut network, hidden_node_id, output_node_id, 2.0);
    /// ```
    pub fn create(
        network: &mut Network,
        node_from_id: usize,
        node_to_id: usize,
        weight: f64,
    ) -> Result<usize> {
        let id = network.edges.iter().map(|e| e.id).max().unwrap_or(0) + 1;

        let node_from = network
            .get_node(node_from_id)
            .context("Node from does not exist")?;

        let node_to = network
            .get_node(node_to_id)
            .context("Node to does not exist")?;

        ensure!(
            node_to.layer_id > node_from.layer_id,
            "node_to must be in a layer after node_from"
        );

        ensure!(
            network.get_edge(id).is_none(),
            "Edge with id {} already exists",
            id
        );

        let edge = Edge {
            id,
            weight,
            node_from_id,
            node_to_id,
        };

        network.edges.push(edge);

        Ok(id)
    }
}
