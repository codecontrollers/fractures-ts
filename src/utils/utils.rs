use ndarray::{Array, Array1, Array2, Order};

use crate::{cost::base::CostModel, wrapper::cost::CostModelTS};

pub fn convert_option<T, U>(input: Option<T>) -> Option<U>
where
  T: TryInto<U>,
  T::Error: std::fmt::Debug,
{
  input.and_then(|x| match x.try_into() {
    Ok(val) => Some(val),
    Err(_) => None,
  })
}

pub fn convert_vector<T, U>(input: Vec<T>) -> Vec<U>
where
  T: TryInto<U>,
  T::Error: std::fmt::Debug,
{
  input
    .into_iter()
    .filter_map(|x| x.try_into().ok())
    .collect()
}

pub fn convert_signal<T: Clone>(input: &Vec<T>) -> Array2<T> {
  let input_array_1d: Array1<T> = Array::from_vec(input.to_owned());
  let input_array_2d: Array2<T> = input_array_1d
    .to_shape(((input_array_1d.dim(), 1), Order::C))
    .unwrap()
    .to_owned();

  input_array_2d
}

pub fn convert_cost_model(input: CostModelTS) -> CostModel {
  match input {
    CostModelTS::AR => CostModel::AR(()),
    CostModelTS::CLinear => CostModel::CLinear(()),
    CostModelTS::Cosine => CostModel::Cosine(()),
    CostModelTS::L1 => CostModel::L1(()),
    CostModelTS::L2 => CostModel::L2(()),
    CostModelTS::Linear => CostModel::Linear(()),
    CostModelTS::Tml => CostModel::Tml(()),
    CostModelTS::Normal => CostModel::Normal(()),
    CostModelTS::Rank => CostModel::Rank(()),
    CostModelTS::RBF => CostModel::RBF(()),
  }
}
