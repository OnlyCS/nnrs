use std::cmp::Ordering;

#[derive(Eq, Debug, Clone, Copy)]
pub enum LayerID {
    InputNode,
    HiddenNode(usize),
    OutputNode,
}

impl Ord for LayerID {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (LayerID::InputNode, LayerID::InputNode) => Ordering::Equal,
            (LayerID::InputNode, _) => Ordering::Less,
            (LayerID::OutputNode, LayerID::OutputNode) => Ordering::Equal,
            (LayerID::OutputNode, _) => Ordering::Greater,
            (LayerID::HiddenNode(x), LayerID::HiddenNode(y)) => x.cmp(y),
            _ => unreachable!(),
        }
    }
}

impl PartialEq for LayerID {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (LayerID::InputNode, LayerID::InputNode) => true,
            (LayerID::OutputNode, LayerID::OutputNode) => true,
            (LayerID::HiddenNode(x), LayerID::HiddenNode(y)) => x == y,
            _ => false,
        }
    }
}

impl PartialOrd for LayerID {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
