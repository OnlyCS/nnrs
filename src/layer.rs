use std::cmp::Ordering;

/// Assigned to a node to indicate which layer it is in.
#[derive(Eq, Debug, Clone, Copy)]
pub enum LayerID {
    /// The input layer.
    InputLayer,

    /// A hidden layer, with an id.
    HiddenLayer(usize),

    /// The output layer.
    OutputLayer,
}

impl Ord for LayerID {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (LayerID::InputLayer, LayerID::InputLayer) => Ordering::Equal,
            (LayerID::InputLayer, _) => Ordering::Less,
            (LayerID::OutputLayer, LayerID::OutputLayer) => Ordering::Equal,
            (LayerID::OutputLayer, _) => Ordering::Greater,
            (LayerID::HiddenLayer(x), LayerID::HiddenLayer(y)) => x.cmp(y),
            (_, LayerID::InputLayer) => Ordering::Greater,
            (_, LayerID::OutputLayer) => Ordering::Less,
        }
    }
}

impl PartialEq for LayerID {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (LayerID::InputLayer, LayerID::InputLayer) => true,
            (LayerID::OutputLayer, LayerID::OutputLayer) => true,
            (LayerID::HiddenLayer(x), LayerID::HiddenLayer(y)) => x == y,
            _ => false,
        }
    }
}

impl PartialOrd for LayerID {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
