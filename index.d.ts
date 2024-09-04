/* tslint:disable */
/* eslint-disable */

/* auto-generated by NAPI-RS */

export const enum CostModelTS {
  AR = 0,
  CLinear = 1,
  Cosine = 2,
  L1 = 3,
  L2 = 4,
  Linear = 5,
  Tml = 6,
  Normal = 7,
  Rank = 8,
  RBF = 9
}
export interface PeltInitParamsTs {
  minSize?: number
  jump?: number
}
export interface PeltPredictParamsTs {
  pen: number
}
export type PeltTS = PeltTs
export declare class PeltTs {
  constructor(params: PeltInitParamsTs, costModel: CostModelTS)
  fitPredict(signal: Array<number>, params: PeltPredictParamsTs): Array<number>
}
