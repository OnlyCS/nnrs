use rand::Rng;

use crate::activationfn::ActivationFn;

pub(crate) trait SelectRandom {
    fn select_random<R>(rng: &mut R) -> Self
    where
        R: Rng;
}

impl SelectRandom for ActivationFn {
    fn select_random<R>(rng: &mut R) -> Self
    where
        R: Rng,
    {
        let random_activation_fn = rng.gen_range(0..5);
        match random_activation_fn {
            0 => ActivationFn::Sigmoid,
            1 => ActivationFn::Tanh,
            2 => ActivationFn::ReLU,
            3 => ActivationFn::Linear,
            4 => ActivationFn::LeakyReLU,
            _ => unreachable!(),
        }
    }
}
