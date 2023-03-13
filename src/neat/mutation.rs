use std::ops::IndexMut;

use anyhow::{bail, Context, Result};
use rand::{rngs::ThreadRng, seq::SliceRandom, Rng};

use crate::{edge::Edge, network::Network, node::Node};

pub(crate) trait Mutations {
    fn add_node_to_existing_layer(&mut self, rng: ThreadRng) -> Result<()>;
    fn add_node_to_new_layer(&mut self, rng: ThreadRng) -> Result<()>;
    fn add_edge(&mut self, rng: ThreadRng) -> Result<()>;
    fn modify_edge(&mut self, rng: ThreadRng) -> Result<()>;
    fn remove_edge(&mut self, rng: ThreadRng) -> Result<()>;
    fn modify_node(&mut self, rng: ThreadRng) -> Result<()>;
    fn remove_node(&mut self, rng: ThreadRng) -> Result<()>;
}

impl Mutations for Network {
    fn add_edge(&mut self, mut rng: ThreadRng) -> Result<()> {
        let node0 = self.nodes.choose(&mut rng).context("No nodes")?;

        let avalable_nodes = self
            .nodes
            .iter()
            .filter(|node| node.layer_id != node0.layer_id)
            .collect::<Vec<_>>();

        let node1 = avalable_nodes
            .choose(&mut rng)
            .context("Network not initiated properly, make sure there is an input and output layer, and at least one node in each")?;

        let weight = rng.gen_range(0.0..=2.0);

        if node0.layer_id < node1.layer_id {
            Edge::create(self, node0.id, node1.id, weight)?;
        } else {
            Edge::create(self, node1.id, node0.id, weight)?;
        }

        Ok(())
    }

    fn add_node_to_existing_layer(&mut self, mut rng: ThreadRng) -> Result<()> {
        let possible_layers = self
            .nodes
            .iter()
            .filter(|node| node.layer_id.is_hidden())
            .map(|node| node.layer_id)
            .collect::<Vec<_>>();

        let layer_id = match possible_layers.choose(&mut rng) {
            Some(layer_id) => *layer_id,
            None => self.add_layer()?,
        };

        let threshold = rng.gen_range(0.0..=1.0);

        Node::create(self, layer_id, threshold)?;

        Ok(())
    }

    fn add_node_to_new_layer(&mut self, mut rng: ThreadRng) -> Result<()> {
        let layer_id = self.add_layer()?;
        let threshold = rng.gen_range(0.0..=1.0);

        Node::create(self, layer_id, threshold)?;

        Ok(())
    }

    fn modify_edge(&mut self, mut rng: ThreadRng) -> Result<()> {
        let edge = match self.edges.choose_mut(&mut rng) {
            Some(edge) => edge,
            None => {
                self.add_edge(rng.clone())?;
                return self.modify_edge(rng);
            }
        };

        let weight = rng.gen_range(0.0..=2.0);
        edge.weight = weight;

        Ok(())
    }

    fn modify_node(&mut self, mut rng: ThreadRng) -> Result<()> {
        let node = self
			.nodes
			.choose_mut(&mut rng)
			.context("Network not initiated properly, make sure there is an input and output layer, and at least one node in each.")?;

        let threshold = rng.gen_range(0.0..=1.0);
        node.threshold = threshold;

        Ok(())
    }

    fn remove_edge(&mut self, mut rng: ThreadRng) -> Result<()> {
        if self.edges.is_empty() {
            return self.add_edge(rng);
        }

        let edge = rng.gen_range(0..self.edges.len());

        self.edges.remove(edge);

        Ok(())
    }

    fn remove_node(&mut self, mut rng: ThreadRng) -> Result<()> {
        if self.nodes.is_empty() {
            bail!("Network not initiated properly, make sure there is an input and output layer, and at least one node in each.")
        }

        if self
            .nodes
            .iter()
            .filter(|node| node.layer_id.is_hidden())
            .count()
            == 0
        {
            return self.add_node_to_new_layer(rng);
        }

        let available_nodes = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.layer_id.is_hidden())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();

        let node_idx = match available_nodes.choose(&mut rng) {
            Some(n) => n,
            None => return Ok(()),
        };

        let node = self.nodes.index_mut(*node_idx);

        for (edge_idx, edge) in (self.edges.clone().iter().enumerate()).rev() {
            if edge.node_from_id == node.id || edge.node_to_id == node.id {
                self.edges.remove(edge_idx);
            }
        }

        self.nodes.remove(*node_idx);

        Ok(())
    }
}
