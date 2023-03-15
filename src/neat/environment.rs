use crate::{activationfn::ActivationFn, network::Network};
use any_range::AnyRange;
use anyhow::{Context, Result};
use rand::rngs::ThreadRng;

use super::mutation::MutableNetwork;

/// Contains an environment for the NEAT algorithm.
pub struct Environment<F: Fn(&mut Network) -> f64> {
    pub(crate) organisms: Vec<Network>,
    pub(crate) generation: usize,
    pub(crate) training_fn: F,
    pub(crate) best_fitness: f64,
    pub(crate) population: usize,
    pub(crate) rng: ThreadRng,
    pub(crate) mutation_rate: usize,
}

impl<F: Fn(&mut Network) -> f64> Environment<F> {
    pub(crate) fn mutate(&mut self) -> Result<()> {
        for organism in &mut self.organisms {
            for _ in 0..self.mutation_rate {
                organism.randomly_mutate(&mut self.rng)?;
            }
        }

        Ok(())
    }

    pub(crate) fn test(&mut self) {
        for organism in &mut self.organisms {
            let fitness = (self.training_fn)(organism);
            organism.fitness = Some(fitness);
        }

        // sort by fitness g->l
        self.organisms
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    }

    pub(crate) fn select_next_gen(&mut self) {
        // choose the top 5% of the population (at front of organisms)
        // clone each one 20 times and add to new organisms

        let top_5_percent = (self.population as f64 * 0.05) as usize;
        let mut new_organisms = Vec::with_capacity(self.population);

        for organism in &self.organisms[..top_5_percent] {
            for _ in 0..20 {
                new_organisms.push(organism.clone());
            }
        }

        self.organisms = new_organisms;
    }

    pub(crate) fn next_gen(&mut self) -> Result<()> {
        self.test();
        self.select_next_gen();
        self.mutate()?;
        self.generation += 1;

        Ok(())
    }

    /// Run the environment until the fitness is within the given range.
    pub fn run<R>(&mut self, fitness_range: R)
    where
        R: Into<AnyRange<f64>>,
    {
        let range: AnyRange<f64> = fitness_range.into();

        while !range.contains(&self.best_fitness) {
            self.next_gen().unwrap();
            self.best_fitness = self.organisms[0].fitness.unwrap();
        }
    }
}

/// Builder for an environment.
pub struct EnvironmentBuilder<F: Fn(&mut Network) -> f64> {
    pub(crate) input_size: Option<usize>,
    pub(crate) output_size: Option<usize>,
    pub(crate) training_fn: Option<F>,
    pub(crate) population: Option<usize>,
    pub(crate) activation_fn: Option<ActivationFn>,
    pub(crate) mutation_rate: Option<usize>,
}

impl<F: Fn(&mut Network) -> f64> EnvironmentBuilder<F> {
    /// Start building an environment.
    pub fn init() -> Self {
        Self {
            input_size: None,
            output_size: None,
            training_fn: None,
            population: None,
            activation_fn: None,
            mutation_rate: None,
        }
    }

    /// Set the input size.
    pub fn input_size(mut self, input_size: usize) -> Self {
        self.input_size = Some(input_size);
        self
    }

    /// Set the output size.
    pub fn output_size(mut self, output_size: usize) -> Self {
        self.output_size = Some(output_size);
        self
    }

    /// Set the activation function.
    pub fn activation_fn(mut self, activation_fn: ActivationFn) -> Self {
        self.activation_fn = Some(activation_fn);
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

    /// Set the mutation rate.
    pub fn mutation_rate(mut self, mutation_rate: usize) -> Self {
        self.mutation_rate = Some(mutation_rate);
        self
    }

    /// Build the environment, returning a `Result`.
    pub fn try_build(self) -> Result<Environment<F>> {
        let input_size = self.input_size.context("Input size not set")?;
        let output_size = self.output_size.context("Output size not set")?;
        let activation_fn = self.activation_fn.context("Activation function not set")?;

        let organisms = vec![
            Network::create(input_size, output_size, activation_fn)
                .context("Failed to create organism")?;
            self.population.context("Population not set")?
        ];

        Ok(Environment {
            organisms,
            generation: 0,
            training_fn: self.training_fn.context("Training function not set")?,
            best_fitness: 0.0,
            population: self.population.context("Population not set")?,
            rng: rand::thread_rng(),
            mutation_rate: self.mutation_rate.context("Mutation rate not set")?,
        })
    }

    /// Build the environment, panicking if there is an error.
    pub fn build(self) -> Environment<F> {
        self.try_build().unwrap()
    }
}
