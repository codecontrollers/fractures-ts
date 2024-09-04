use std::rc::Rc;

use ndarray::Array2;

use super::costl1::CostL1;

#[derive(Clone, Copy)]
pub enum CostModel {
  AR(()),
  CLinear(()),
  Cosine(()),
  L1(()),
  L2(()),
  Linear(()),
  Tml(()),
  Normal(()),
  Rank(()),
  RBF(()),
}

pub trait CostParam {}

impl CostParam for () {}

pub trait BaseCost<P: CostParam> {
  fn new(params: P) -> Rc<dyn BaseParametrizedCost>
  where
    Self: Sized;
}

pub trait BaseParametrizedCost {
  fn fit(&self, signal: Array2<f64>) -> Rc<dyn BaseFittedCost>;
  fn min_size(&self) -> usize;
}

pub trait BaseFittedCost {
  fn error(&self, start: usize, end: usize) -> Option<f64>;
}

pub fn cost_factory(cost_params: CostModel) -> Rc<Rc<dyn BaseParametrizedCost>> {
  match cost_params {
    CostModel::AR(_) => todo!(),
    CostModel::CLinear(_) => todo!(),
    CostModel::Cosine(_) => todo!(),
    CostModel::L1(x) => Rc::new(CostL1::new(x)),
    CostModel::L2(_) => todo!(),
    CostModel::Linear(_) => todo!(),
    CostModel::Tml(_) => todo!(),
    CostModel::Normal(_) => todo!(),
    CostModel::Rank(_) => todo!(),
    CostModel::RBF(_) => todo!(),
  }
}
