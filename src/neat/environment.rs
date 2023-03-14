use crate::network::Network;
use anyhow::{Context, Result};

/// Contains an environment for the NEAT algorithm.
pub struct Environment<F: Fn(&mut Network) -> f64> {
    pub(crate) organisms: Vec<Network>,
    pub(crate) generation: usize,
    pub(crate) training_fn: F,
    pub(crate) best_fitness: f64,
    pub(crate) population: usize,
}

/// Builder for an environment.
pub struct EnvironmentBuilder<F: Fn(&mut Network) -> f64> {
    pub(crate) organism: Option<Network>,
    pub(crate) training_fn: Option<F>,
    pub(crate) population: Option<usize>,
}

impl<F: Fn(&mut Network) -> f64> EnvironmentBuilder<F> {
    /// Start building an environment.
    pub fn init() -> Self {
        Self {
            organism: None,
            training_fn: None,
            population: None,
        }
    }

    /// Set the starting organism.
    pub fn starting_organism(mut self, organism: Network) -> Self {
        self.organism = Some(organism);
        self
    }

    /// Set the training function.
    pub fn training_fn(mut self, training_fn: F) -> Self {
        self.training_fn = Some(training_fn);
        self
    }

    /// Set the population size.
    pub fn population(mut self, population: usize) -> Self {
        self.population = Some(population);
        self
    }

    /// Build the environment, returning a `Result`.
    pub fn try_build(self) -> Result<Environment<F>> {
        let organisms = vec![
            self.organism.context("Starting organism not set")?;
            self.population.context("Population not set")?
        ];

        Ok(Environment {
            organisms,
            generation: 0,
            training_fn: self.training_fn.context("Training function not set")?,
            best_fitness: 0.0,
            population: self.population.context("Population not set")?,
        })
    }

    /// Build the environment, panicking if there is an error.
    pub fn build(self) -> Environment<F> {
        self.try_build().unwrap()
    }
}
