package org.mengpan.deeplearning.components.initializer

import breeze.linalg.DenseMatrix
import breeze.numerics.sqrt

/**
  * Created by mengpan on 2017/9/5.
  */
object XaiverInitializer extends WeightsInitializer{
  private def getWeightsMultipliyer(previousLayerDim: Int): Double = {
    sqrt(1.0/previousLayerDim)
  }

  override def init(rows: Int, cols: Int)(implicit getWeightsMultipliyer: (Int) => Double): DenseMatrix[Double] = super.init(rows, cols)(this.getWeightsMultipliyer)
}
