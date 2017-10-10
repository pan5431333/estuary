package estuary.components.initializer

import breeze.linalg.DenseMatrix

/**
  * Created by mengpan on 2017/9/5.
  */
object NormalInitializer extends WeightsInitializer {
  override def init(rows: Int, cols: Int)(implicit getWeightsMultipliyer: (Int) => Double): DenseMatrix[Double] = super.init(rows, cols)(x => 0.01)
}
