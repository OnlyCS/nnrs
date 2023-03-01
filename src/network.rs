use crate::{edge::Edge, layer::LayerID, node::Node};

#[derive(Clone)]
pub struct Network {
    pub(crate) nodes: Vec<Node>,
    pub(crate) edges: Vec<Edge>,
}

impl Network {
    pub fn get_node(&self, node_id: usize) -> Option<&Node> {
        self.nodes.iter().find(|node| node.id == node_id)
    }

    pub fn get_layer(&self, layer_id: LayerID) -> Option<Vec<&Node>> {
        let matches = self
            .nodes
            .iter()
            .filter(|node| node.layer_id == layer_id)
            .collect::<Vec<&Node>>();

        if matches.is_empty() {
            None
        } else {
            Some(matches)
        }
    }

    pub fn get_edge(&self, edge_id: usize) -> Option<&Edge> {
        self.edges.iter().find(|edge| edge.id == edge_id)
    }
}
