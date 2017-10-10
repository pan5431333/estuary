package estuary.components.initializer

import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Rand

/**
  * Created by mengpan on 2017/9/5.
  */
trait WeightsInitializer {
  def init(rows: Int, cols: Int)(implicit getWeightsMultipliyer: Int => Double): DenseMatrix[Double] = {
    DenseMatrix.rand[Double](rows, cols, rand = Rand.gaussian) * getWeightsMultipliyer(rows)
  }
}
