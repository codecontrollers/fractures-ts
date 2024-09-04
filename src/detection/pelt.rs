use std::{collections::HashMap, rc::Rc};

use ndarray::Array2;

use crate::cost::base::{cost_factory, BaseFittedCost, BaseParametrizedCost, CostModel};

use super::base::{
  BaseDetection, BaseFittedDetection, BaseParametrizedDetection, DtcInitParam, DtcParams,
  DtcPredictParam,
};

pub struct PeltInitParams {
  pub min_size: Option<usize>,
  pub jump: Option<usize>,
}

impl DtcInitParam for PeltInitParams {}

pub struct PeltPredictParams {
  pub pen: f64,
}

impl DtcPredictParam for PeltPredictParams {}

pub struct PeltParams {}

impl DtcParams for PeltParams {
  type Init = PeltInitParams;
  type Predict = PeltPredictParams;
}

#[derive(Clone)]
pub struct Pelt {
  cost_model_instance: Rc<Rc<dyn BaseParametrizedCost>>,
  min_size: usize,
  jump: usize,
}

pub struct FittedPelt {
  previous: Pelt,
  fitted_cost_model_instance: Rc<dyn BaseFittedCost>,
  n_samples: usize,
}

impl BaseDetection<PeltParams> for Pelt {
  fn new(
    cost_model: CostModel,
    params: PeltInitParams,
  ) -> Rc<dyn BaseParametrizedDetection<PeltParams>> {
    let cost_model_instance = cost_factory(cost_model);
    let min_size = usize::max(params.min_size.unwrap_or(2), cost_model_instance.min_size());
    let jump = params.jump.unwrap_or(5);

    Rc::new(Pelt {
      cost_model_instance: cost_model_instance,
      min_size,
      jump,
    })
  }
}

impl BaseParametrizedDetection<PeltParams> for Pelt {
  fn fit(&self, signal: Array2<f64>) -> Rc<dyn BaseFittedDetection<PeltParams>> {
    let (n_samples, _) = signal.dim();
    let fitted_cost_model_instance = self.cost_model_instance.fit(signal);

    Rc::new(FittedPelt {
      previous: self.clone(),
      fitted_cost_model_instance: fitted_cost_model_instance,
      n_samples,
    })
  }
}

impl BaseFittedDetection<PeltParams> for FittedPelt {
  fn predict(&self, params: PeltPredictParams) -> Vec<usize> {
    let partition = self.seg(params.pen);

    // Extracting and sorting the end points from the partition keys
    let mut bkps: Vec<usize> = partition.keys().map(|&(_, e)| e).collect();
    bkps.sort_unstable();

    bkps
  }
}

impl FittedPelt {
  fn seg(&self, pen: f64) -> HashMap<(usize, usize), f64> {
    let mut admissible: Vec<usize> = vec![];
    let mut partitions: HashMap<usize, HashMap<(usize, usize), f64>> = HashMap::new();

    partitions.insert(0, {
      let mut inner_map = HashMap::new();
      inner_map.insert((0, 0), 0.0);
      inner_map
    });

    let mut ind: Vec<usize> = (0..self.n_samples)
      .step_by(self.previous.jump)
      .filter(|&k| k >= self.previous.min_size)
      .collect();

    ind.push(self.n_samples);

    for bkp in ind {
      let new_adm_pt = bkp - self.previous.min_size;
      admissible.push(new_adm_pt);

      let mut subproblems: Vec<HashMap<(usize, usize), f64>> = vec![];

      for &t in &admissible {
        // Try to get a copy of the partition from the partitions HashMap
        if let Some(tmp_partition) = partitions.get(&t).cloned() {
          let mut tmp_partition = tmp_partition.clone();
          // Update the partition with the right partition
          tmp_partition.insert(
            (t, bkp),
            self.fitted_cost_model_instance.error(t, bkp).unwrap() + pen,
          );
          // Append the updated partition to the subproblems vector
          subproblems.push(tmp_partition);
        }
      }

      // Finding the optimal partition
      if let Some(optimal_partition) = subproblems.iter().min_by(|a, b| {
        a.values()
          .sum::<f64>()
          .partial_cmp(&b.values().sum::<f64>())
          .unwrap()
      }) {
        partitions.insert(bkp, optimal_partition.clone());
      }

      // Trimming the admissible set
      admissible = admissible
        .into_iter()
        .zip(subproblems.into_iter())
        .filter(|(_, partition)| {
          partition.values().sum::<f64>() <= partitions[&bkp].values().sum::<f64>() + pen
        })
        .map(|(t, _)| t)
        .collect();
    }

    // Retrieve the best partition
    let mut best_partition = partitions.remove(&self.n_samples).unwrap();
    // Remove the entry (0, 0)
    best_partition.remove(&(0, 0));
    // Return the best partition
    best_partition
  }
}
