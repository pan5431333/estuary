package estuary.components.initializer

import breeze.linalg.DenseMatrix
import breeze.numerics.sqrt

/**
  * Created by mengpan on 2017/9/5.
  */
object HeInitializer extends WeightsInitializer {
  private def getWeightsMultipliyer(previousLayerDim: Int): Double = {
    sqrt(2.0 / previousLayerDim)
  }

  override def init(rows: Int, cols: Int)(implicit getWeightsMultipliyer: (Int) => Double): DenseMatrix[Double] = super.init(rows, cols)(this.getWeightsMultipliyer)
}
