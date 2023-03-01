use crate::network::Network;
use anyhow::{ensure, Context, Result};

#[derive(Clone)]
pub struct Edge {
    pub(crate) id: usize,
    pub(crate) weight: f64,
    pub(crate) node_from_id: usize,
    pub(crate) node_to_id: usize,
}

impl Edge {
    pub fn create(
        network: &mut Network,
        id: usize,
        node_from_id: usize,
        node_to_id: usize,
        weight: f64,
    ) -> Result<()> {
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

        Ok(())
    }
}
