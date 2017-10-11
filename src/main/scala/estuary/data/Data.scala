package estuary.data

import breeze.linalg.DenseVector

/**
  * Created by mengpan on 2017/8/24.
  */
trait Data {
  var feature: DenseVector[Double]

  var label: Double

  def updateFeature(feature: DenseVector[Double]): this.type = {
    this.feature = feature
    this
  }
}
