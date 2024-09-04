use std::rc::Rc;

use ndarray::{s, Array2, Axis};
use ndarray_stats::{interpolate::Midpoint, QuantileExt};
use noisy_float::types::n64;

use super::base::{BaseCost, BaseFittedCost, BaseParametrizedCost};

#[derive(Clone)]
pub struct CostL1 {
  pub min_size: usize,
}

pub struct FittedCostL1 {
  previous: CostL1,
  signal: Array2<f64>,
}

impl BaseCost<()> for CostL1 {
  fn new(_params: ()) -> Rc<dyn BaseParametrizedCost> {
    let min_size: usize = 2;

    Rc::new(CostL1 { min_size })
  }
}

impl BaseParametrizedCost for CostL1 {
  fn fit(&self, signal: Array2<f64>) -> Rc<dyn BaseFittedCost> {
    Rc::new(FittedCostL1 {
      previous: self.clone(),
      signal,
    })
  }
  fn min_size(&self) -> usize {
    self.min_size
  }
}

impl BaseFittedCost for FittedCostL1 {
  fn error(&self, start: usize, end: usize) -> Option<f64> {
    if end - start < self.previous.min_size {
      return None;
    }

    let mut sub = self.signal.slice(s![start..end, ..]).to_owned();

    let med = sub
      .quantile_axis_skipnan_mut(Axis(0), n64(0.5), &Midpoint)
      .unwrap();
    let diff = sub - med;
    let abs_diff = diff.abs();
    let abs_diff_sum = abs_diff.sum();

    Some(abs_diff_sum)
  }
}

#[cfg(test)]
mod tests {
  use crate::{
    cost::base::{cost_factory, CostModel},
    utils::utils::convert_signal,
  };

  #[test]
  fn test_costl1_new() {
    let test = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let test_2d = convert_signal(&test);

    let cost = cost_factory(CostModel::L1(()));
    let fitted_cost = cost.fit(test_2d);
    fitted_cost.error(0, 1);
  }
}
