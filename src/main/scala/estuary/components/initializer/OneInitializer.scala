package estuary.components.initializer

import breeze.linalg.DenseMatrix

object OneInitializer extends WeightsInitializer{
  override def init(rows: Int, cols: Int)(implicit getWeightsMultipliyer: (Int) => Double) = {
    DenseMatrix.ones[Double](rows, cols)
  }
}
