use std::rc::Rc;

use ndarray::Array2;

use crate::cost::base::CostModel;

pub enum DetectionModel {
  BinSeg(),
  BottomUp(),
  DynP(),
  KernelCPD(),
  Pelt(),
  Window(),
}

pub trait DtcInitParam {}

impl DtcInitParam for () {}

pub trait DtcPredictParam {}

impl DtcPredictParam for () {}

pub trait DtcParams {
  type Init: DtcInitParam;
  type Predict: DtcPredictParam;
}

pub trait BaseDetection<T: DtcParams> {
  fn new(cost_model: CostModel, params: T::Init) -> Rc<dyn BaseParametrizedDetection<T>>
  where
    Self: Sized;
}

pub trait BaseParametrizedDetection<T: DtcParams> {
  fn fit(&self, signal: Array2<f64>) -> Rc<dyn BaseFittedDetection<T>>;
}

pub trait BaseFittedDetection<T: DtcParams> {
  fn predict(&self, params: T::Predict) -> Vec<usize>;
}
