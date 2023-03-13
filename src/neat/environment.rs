use std::ops::IndexMut;

use super::{
    mutation::Mutations,
    settings::{MutationType, Settings, TrainingMode},
};
use crate::network::Network;
use anyhow::{Context, Result};
use rand::{
    rngs::ThreadRng,
    seq::{IteratorRandom, SliceRandom},
    thread_rng, Rng,
};

/// The NEAT environment.
///
/// # Examples
/// ```
/// use nnrs::{neat::{environment::Environment, settings::Settings}, network::Network, node::Node, edge::Edge, layer::LayerID};
///
/// // xor classification
/// let mut network = Network::default();
///
/// Node::create(&mut network, LayerID::InputLayer, 0.0).unwrap();
/// Node::create(&mut network, LayerID::InputLayer, 0.0).unwrap();
/// Node::create(&mut network, LayerID::OutputLayer, 0.0).unwrap();
///
/// let mut environment = Environment::new(Settings::default(), network, |network| {
///     let mut distance = 0.0;
///     let mut output = vec![];
///
///     network.set_inputs(vec![0.0, 0.0]).unwrap();
///     network.fire().unwrap();
///
///     network.read(&mut output).unwrap();
///     distance += (output[0] - 0.0).powi(2);
///
///     network.reset();
///     output.clear();
///
///     network.set_inputs(vec![0.0, 1.0]).unwrap();
///     network.fire().unwrap();
///
///     network.read(&mut output).unwrap();
///     distance += (output[0] - 1.0).powi(2);
///
///     network.reset();
///     output.clear();
///
///     network.set_inputs(vec![1.0, 0.0]).unwrap();
///     network.fire().unwrap();
///
///     network.read(&mut output).unwrap();
///     distance += (output[0] - 1.0).powi(2);
///
///     network.reset();
///     output.clear();
///
///     network.set_inputs(vec![1.0, 1.0]).unwrap();
///     network.fire().unwrap();
///
///     network.read(&mut output).unwrap();
///     distance += (output[0] - 0.0).powi(2);
///
///     network.reset();
///
///     1.0 / (1.0 + distance)
/// });
///
/// let mut champion: Network = environment.run().unwrap();
/// ```
pub struct Environment<T: Fn(&mut Network) -> f64> {
    pub(crate) population: Vec<Network>,
    pub(crate) evaluator: T,
    pub(crate) generation: usize,
    pub(crate) settings: Settings,
    pub(crate) rng: ThreadRng,
}

impl<T: Fn(&mut Network) -> f64> Environment<T> {
    /// Creates a new NEAT environment.
    ///
    /// ### Example
    /// ```
    /// use nnrs::{neat::{environment::Environment, settings::Settings}, network::Network, node::Node, edge::Edge, layer::LayerID};
    ///
    /// let mut network = Network::default();
    ///
    /// Node::create(&mut network, LayerID::InputLayer, 0.0).unwrap();
    /// Node::create(&mut network, LayerID::InputLayer, 0.0).unwrap();
    /// Node::create(&mut network, LayerID::OutputLayer, 0.0).unwrap();
    ///
    /// let mut environment = Environment::new(Settings::default(), network, |network| {
    ///     // evaluation function. returns f64
    ///     5.0
    /// });
    /// ```
    pub fn new(settings: Settings, starting_network: Network, evaluator: T) -> Self {
        Self {
            population: vec![starting_network; settings.population_size],
            evaluator,
            generation: 0,
            settings,
            rng: thread_rng(),
        }
    }

    pub(crate) fn randomly_mutate(&mut self, network: &mut Network) -> Result<()> {
        let num_mutations = self.rng.gen_range(self.settings.mutation_rate.clone());

        for _ in 0..num_mutations {
            let mutation_type = self
                .settings
                .mutation_types
                .choose(&mut self.rng)
                .context("No mutation types available")?;

            match mutation_type {
                MutationType::AddNodeToExistingLayer => {
                    network.add_node_to_existing_layer(self.rng.clone())?
                }
                MutationType::AddNodeToNewLayer => {
                    network.add_node_to_new_layer(self.rng.clone())?
                }
                MutationType::AddEdge => network.add_edge(self.rng.clone())?,
                MutationType::ModifyEdge => network.modify_edge(self.rng.clone())?,
                MutationType::RemoveEdge => network.remove_edge(self.rng.clone())?,
                MutationType::ModifyNode => network.modify_node(self.rng.clone())?,
                MutationType::RemoveNode => network.remove_node(self.rng.clone())?,
            }
        }

        Ok(())
    }

    pub(crate) fn evaluate(&mut self, index: usize) -> Result<()> {
        let network = self.population.index_mut(index);
        let fitness = (self.evaluator)(network);
        network.fitness = Some(fitness);

        Ok(())
    }

    pub(crate) fn evaluate_all(&mut self) -> Result<()> {
        for index in 0..self.population.len() {
            self.evaluate(index)?;
        }

        Ok(())
    }

    pub(crate) fn sort_by_fitness(&mut self) {
        self.population.sort_by(|a, b| {
            b.fitness
                .unwrap_or_default()
                .total_cmp(&a.fitness.unwrap_or_default())
        });
    }

    pub(crate) fn next_generation(&mut self) -> Result<()> {
        let mut mutated = vec![];

        for index in 0..self.settings.next_gen_threshold {
            let mut population = self.population.clone();

            for _ in 0..(self.settings.population_size / self.settings.next_gen_threshold) {
                let network = population.index_mut(index);

                for _ in 0..self
                    .settings
                    .mutation_rate
                    .clone()
                    .choose(&mut self.rng)
                    .unwrap()
                {
                    self.randomly_mutate(network)?;
                }

                mutated.push(network.clone());
            }
        }

        self.population = mutated;

        self.evaluate_all()?;
        self.sort_by_fitness();

        println!("max: {}", self.population[0].fitness.unwrap_or_default());

        Ok(())
    }

    /// Runs the environment until the training requirement is satisfied. Returns the best network.
    ///
    /// ### Example
    /// ```
    /// use nnrs::{neat::{environment::Environment, settings::Settings}, network::Network, node::Node, edge::Edge, layer::LayerID};
    ///
    /// let mut network = Network::default();
    /// Node::create(&mut network, LayerID::InputLayer, 0.0).unwrap();
    /// Node::create(&mut network, LayerID::InputLayer, 0.0).unwrap();
    /// Node::create(&mut network, LayerID::OutputLayer, 0.0).unwrap();
    ///
    /// let mut environment = Environment::new(Settings::default(), network, |network| {
    ///    // evaluation function. returns f64
    ///    5.0
    /// });
    ///
    /// let mut champion: Network = environment.run().unwrap();
    /// ```
    pub fn run(&mut self) -> Result<Network> {
        loop {
            self.next_generation()?;
            self.generation += 1;

            match self.settings.training_mode.clone() {
                TrainingMode::FitnessTarget(x) => {
                    if let Some(fitness) = self.population[0].fitness {
                        if fitness >= x {
                            break;
                        }
                    }
                }
                TrainingMode::NumGenerations(x) => {
                    if self.generation >= x {
                        break;
                    }
                }
            }
        }

        Ok(self.population[0].clone())
    }
}
