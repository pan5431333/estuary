package org.mengpan.deeplearning.components.initializer

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Rand
import org.mengpan.deeplearning.components.layers.Layer

/**
  * Created by mengpan on 2017/9/5.
  */
trait WeightsInitializer {
  def init(rows: Int, cols: Int)(implicit getWeightsMultipliyer: Int => Double): DenseMatrix[Double] = {
    DenseMatrix.rand[Double](rows, cols, rand=Rand.gaussian) * getWeightsMultipliyer(rows)
  }
}
