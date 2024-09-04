import test from "ava"

import { PeltTs, CostModelTS } from "../index.js"

test("sum from native", (t) => {
  let signal = [
    0.2, 144.0, 488.0, 517.0, 533.0, 533.0, 533.0, 533.0, 528.0, 472.0, 490.0,
    535.0, 537.0, 549.0, 631.0,
  ]

  let pelt = new PeltTs({ minSize: 1, jump: 1 }, CostModelTS.L1)

  let result = pelt.fitPredict(signal, { pen: 1.05 })

  console.log(result)
})
