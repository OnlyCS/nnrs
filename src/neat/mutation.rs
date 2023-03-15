use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
};

use crate::{activationfn::ActivationFn, edge::Edge, layer::LayerID, network::Network, node::Node};
use anyhow::Result;

use super::random::SelectRandom;

pub(crate) enum Mutation {
    AddNode,
    AddLayer,
    AddEdge,
    RemoveNode,
    RemoveEdge,
    ChangeWeight,
    ChangeBias,
    ChangeActivationFn,
}

impl SelectRandom for Mutation {
    fn select_random<R>(rng: &mut R) -> Self
    where
        R: Rng,
    {
        let random_mutation = rng.gen_range(0..8);
        match random_mutation {
            0 => Mutation::AddNode,
            1 => Mutation::AddLayer,
            2 => Mutation::AddEdge,
            3 => Mutation::RemoveNode,
            4 => Mutation::RemoveEdge,
            5 => Mutation::ChangeWeight,
            6 => Mutation::ChangeBias,
            7 => Mutation::ChangeActivationFn,
            _ => unreachable!(),
        }
    }
}

pub(crate) trait MutableNetwork {
    fn mutate(&mut self, mutation: Mutation) -> Result<()>;
    fn randomly_mutate<R>(&mut self, rng: &mut R) -> Result<()>
    where
        R: Rng,
    {
        let mutation = Mutation::select_random(rng);
        self.mutate(mutation)
    }
}

impl MutableNetwork for Network {
    fn mutate(&mut self, mutation: Mutation) -> Result<()> {
        let mut rng = rand::thread_rng();

        match mutation {
            Mutation::AddNode => {
                // iterate over all non input and non output layers
                // pick a random layer
                // if none exists, call this function again with AddLayer mutation
                // add a node with the default activation function, random bias and threshold

                let layers = self
                    .layers
                    .iter()
                    .filter(|l| l.is_hidden())
                    .copied()
                    .collect::<Vec<LayerID>>();

                let layer = match layers.choose(&mut rng) {
                    Some(x) => *x,
                    None => {
                        return self.mutate(Mutation::AddLayer);
                    }
                };

                Node::create(self, layer, rng.gen_range(-1.0..1.0))?;
            }
            Mutation::AddLayer => {
                // create a layer and add a node to it

                let layer = self.add_layer();

                Node::create(self, layer, rng.gen_range(-1.0..1.0))?;
            }
            Mutation::AddEdge => {
                // pick two random nodes, the first from any layer and the second from a layer not equal to the first
                // if no such nodes exist, call this function again with AddNode mutation
                // add an edge between the two nodes
                // if the edge already exists, call this function again with the ChangeWeight mutation

                let node1 = match self.nodes.choose(&mut rng) {
                    Some(x) => x,
                    None => {
                        return self.mutate(Mutation::AddNode);
                    }
                };

                let node2 = match self
                    .nodes
                    .iter()
                    .filter(|n| n.layer_id != node1.layer_id)
                    .choose(&mut rng)
                {
                    Some(x) => x,
                    None => {
                        return self.mutate(Mutation::AddNode);
                    }
                };

                let (node_begin, node_end) = if node1.layer_id < node2.layer_id {
                    (node1, node2)
                } else {
                    (node2, node1)
                };

                if self
                    .edges
                    .iter()
                    .any(|e| e.node_from_id == node_begin.id && e.node_to_id == node_end.id)
                {
                    return self.mutate(Mutation::ChangeWeight);
                }

                Edge::create(self, node_begin.id, node_end.id, rng.gen_range(-1.0..1.0))?;
            }
            Mutation::RemoveNode => {
                // pick a random, not input or output node
                // if none exists, call this function again with AddNode mutation
                // remove the node and all edges connected to it

                let (node_index, node) = match self
                    .nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| n.layer_id.is_hidden())
                    .choose(&mut rng)
                {
                    Some(x) => x,
                    None => {
                        return self.mutate(Mutation::AddNode);
                    }
                };

                self.edges
                    .retain(|e| e.node_from_id != node.id && e.node_to_id != node.id);
                self.nodes.remove(node_index);
            }
            Mutation::RemoveEdge => {
                // pick a random edge
                // if none exists, call this function again with AddEdge mutation
                // remove the edge

                if self.edges.is_empty() {
                    return self.mutate(Mutation::AddEdge);
                }

                let edge_index = rng.gen_range(0..self.edges.len());
                self.edges.remove(edge_index);
            }
            Mutation::ChangeWeight => {
                // pick a random edge
                // if none exists, call this function again with AddEdge mutation
                // change the weight of the edge

                let edge = match self.edges.choose_mut(&mut rng) {
                    Some(x) => x,
                    None => {
                        return self.mutate(Mutation::AddEdge);
                    }
                };

                edge.weight += rng.gen_range(-1.0..1.0);
            }
            Mutation::ChangeBias => {
                // pick a random node
                // if none exists, call this function again with AddNode mutation
                // change the bias of the node

                let node = match self.nodes.choose_mut(&mut rng) {
                    Some(x) => x,
                    None => {
                        return self.mutate(Mutation::AddNode);
                    }
                };

                node.bias += rng.gen_range(-1.0..1.0);
            }
            Mutation::ChangeActivationFn => {
                // pick a random node
                // if none exists, call this function again with AddNode mutation
                // change the activation function of the node

                let node = match self.nodes.choose_mut(&mut rng) {
                    Some(x) => x,
                    None => {
                        return self.mutate(Mutation::AddNode);
                    }
                };

                if node.layer_id == LayerID::OutputLayer {
                    node.activation_fn = ActivationFn::Binary(rng.gen_range(0.0..1.0));
                }

                node.activation_fn = ActivationFn::select_random(&mut rng);
            }
        }

        Ok(())
    }
}
