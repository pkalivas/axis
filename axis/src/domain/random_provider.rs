use rand::distributions::Standard;
use rand::distributions::{uniform::SampleUniform, Distribution};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use std::sync::{Arc, Mutex, OnceLock};

struct RandomProvider {
    rng: Arc<Mutex<StdRng>>,
}

impl RandomProvider {
    /// Returns the global instance of the registry.
    pub(self) fn global() -> &'static RandomProvider {
        static INSTANCE: OnceLock<RandomProvider> = OnceLock::new();

        INSTANCE.get_or_init(|| RandomProvider {
            rng: Arc::new(Mutex::new(StdRng::from_entropy())),
        })
    }

    pub(self) fn replace_global(other: StdRng) {
        let instance = RandomProvider::global();
        let mut rng = instance.rng.lock().unwrap();
        *rng = other;
    }

    pub(self) fn get_global() -> StdRng {
        let instance = RandomProvider::global();
        let rng = instance.rng.lock().unwrap();
        rng.clone()
    }

    /// Sets a new seed for the global RNG.
    pub(self) fn set_seed(seed: u64) {
        let instance = RandomProvider::global();
        let mut rng = instance.rng.lock().unwrap();
        *rng = StdRng::seed_from_u64(seed);
    }

    /// Generates a random number using the global RNG.
    pub(self) fn random<T>() -> T
    where
        T: rand::distributions::uniform::SampleUniform,
        Standard: Distribution<T>,
    {
        let instance = RandomProvider::global();
        let mut rng = instance.rng.lock().unwrap();
        rng.gen()
    }

    pub(self) fn gen_range<T>(range: std::ops::Range<T>) -> T
    where
        T: SampleUniform + PartialOrd,
        Standard: Distribution<T>,
    {
        let instance = RandomProvider::global();
        let mut rng = instance.rng.lock().unwrap();
        rng.gen_range(range)
    }
}

/// Seeds the thread-local random number generator with the given seed.
pub fn set_seed(seed: u64) {
    RandomProvider::set_seed(seed);
}

/// Generates a random number of type T.
///
/// For floating point types, the number will be in the range [0, 1).
/// For integer types, the number will be in the range [0, MAX).
pub fn random<T>() -> T
where
    T: SampleUniform,
    Standard: Distribution<T>,
{
    RandomProvider::random()
}

/// Generates a random number of type T in the given range.
pub fn gen_range<T>(range: std::ops::Range<T>) -> T
where
    T: SampleUniform + PartialOrd,
    Standard: Distribution<T>,
{
    RandomProvider::gen_range(range)
}

/// Chooses a random item from the given slice.
pub fn choose<T>(items: &[T]) -> &T {
    let index = gen_range(0..items.len());
    &items[index]
}

/// Generates a random number from a Gaussian distribution with the given mean and standard deviation.
/// The Box-Muller transform is used to generate the random number.
pub fn gaussian(mean: f64, std_dev: f64) -> f64 {
    let u1: f64 = RandomProvider::random();
    let u2: f64 = RandomProvider::random();

    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

    mean + std_dev * z0
}

/// Shuffles the given slice in place.
pub fn shuffle<T>(items: &mut [T]) {
    let instance = RandomProvider::global();
    items.shuffle(&mut *instance.rng.lock().unwrap());
}

/// Generates a vector of indexes from 0 to n-1 in random order.
pub fn indexes(n: usize) -> Vec<usize> {
    let mut indexes: Vec<usize> = (0..n).collect();
    shuffle(&mut indexes);
    indexes
}

pub fn scoped_seed<F>(seed: u64, func: F)
where
    F: FnOnce(),
{
    let current_rng = RandomProvider::get_global();

    RandomProvider::replace_global(StdRng::seed_from_u64(seed));
    func();
    RandomProvider::replace_global(current_rng);
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_random() {
        for _ in 0..100 {
            let value: f64 = random();
            assert!(value >= 0.0 && value < 1.0);
        }
    }

    #[test]
    fn test_gen_range() {
        for _ in 0..100 {
            let value: f64 = gen_range(0.0..100.0);
            assert!(value >= 0.0 && value < 100.0);
        }
    }

    #[test]
    fn test_choose() {
        for _ in 0..100 {
            let items = vec![1, 2, 3, 4, 5];
            let value = choose(&items);
            assert!(items.contains(value));
        }
    }

    #[test]
    fn test_shuffle() {
        let mut items = vec![1, 2, 3, 4, 5];
        shuffle(&mut items);
        assert_ne!(items, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_indexes() {
        let indexes = indexes(10);
        assert_eq!(indexes.len(), 10);
        assert_ne!(indexes, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
