use std::ops::RangeInclusive;

/// The way the environment knows when to stop training
#[derive(Debug, Clone)]
pub enum TrainingMode {
    /// A target fitness value, defined by evaluator function. Recommended.
    FitnessTarget(f64),

    /// A number of generations to train for.
    NumGenerations(usize),
}

/// The type of mutation to perform
#[derive(Debug, Clone, Copy)]
pub enum MutationType {
    /// Add a node to an existing layer
    AddNodeToExistingLayer,

    /// Add a node to a new layer
    AddNodeToNewLayer,

    /// Add an edge between two nodes
    AddEdge,

    /// Modify an edge's weight
    ModifyEdge,

    /// Remove an edge
    RemoveEdge,

    /// Modify a node's threshold
    ModifyNode,

    /// Remove a node
    RemoveNode,
}

/// A struct containing all the settings for the environment
#[derive(Debug, Clone)]
pub struct Settings {
    /// The size of the population
    pub population_size: usize,

    /// The types of mutations to perform. Including one twice will increase the chance of it being chosen.
    pub mutation_types: Vec<MutationType>,

    /// The way the environment knows when to stop training
    pub training_mode: TrainingMode,

    /// The range of mutations to perform. The number of mutations is chosen randomly from this range.
    pub mutation_rate: RangeInclusive<usize>,

    /// The number of orgaisms kept from the previous generation to be used in the next generation.
    pub next_gen_threshold: usize,
}

impl Default for Settings {
    fn default() -> Self {
        // percent chance for mutation types should be as follows:
        // 15% AddNodeToExistingLayer
        // 5% AddNodeToNewLayer
        // 10% AddEdge
        // 30% ModifyEdge
        // 30% ModifyNode
        // 5% RemoveEdge
        // 5% RemoveNode

        let mut mut_types = vec![];

        mut_types.extend([MutationType::AddNodeToExistingLayer; 3]);
        mut_types.extend([MutationType::AddNodeToNewLayer; 1]);
        mut_types.extend([MutationType::AddEdge; 2]);
        mut_types.extend([MutationType::ModifyEdge; 6]);
        mut_types.extend([MutationType::ModifyNode; 6]);
        mut_types.extend([MutationType::RemoveEdge; 1]);
        mut_types.extend([MutationType::RemoveNode; 1]);

        Self {
            population_size: 100,
            mutation_types: mut_types,
            training_mode: TrainingMode::NumGenerations(30),
            mutation_rate: 0..=5,
            next_gen_threshold: 5,
        }
    }
}
