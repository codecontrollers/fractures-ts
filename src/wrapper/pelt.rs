use std::rc::Rc;

use crate::{
  detection::{
    base::{BaseDetection, BaseParametrizedDetection},
    pelt::{Pelt, PeltInitParams, PeltParams, PeltPredictParams},
  },
  utils::utils::{convert_cost_model, convert_option, convert_signal, convert_vector},
};

use super::cost::CostModelTS;

#[napi(object)]
pub struct PeltInitParamsTS {
  pub min_size: Option<i64>,
  pub jump: Option<i64>,
}

#[napi(object)]
pub struct PeltPredictParamsTS {
  pub pen: f64,
}

#[napi]
pub struct PeltTS {
  detector_model_instance: Rc<dyn BaseParametrizedDetection<PeltParams>>,
}

#[napi]
impl PeltTS {
  #[napi(constructor)]
  pub fn new(params: PeltInitParamsTS, cost_model: CostModelTS) -> Self {
    let cost_model = convert_cost_model(cost_model);

    let params = PeltInitParams {
      min_size: convert_option(params.min_size),
      jump: convert_option(params.jump),
    };

    let detector_model_instance = Pelt::new(cost_model, params);

    PeltTS {
      detector_model_instance,
    }
  }

  #[napi]
  pub fn fit_predict(&self, signal: Vec<f64>, params: PeltPredictParamsTS) -> Vec<i64> {
    let signal = convert_signal(&signal);

    let params = PeltPredictParams { pen: params.pen };

    let result = self.detector_model_instance.fit(signal).predict(params);

    convert_vector(result)
  }
}
