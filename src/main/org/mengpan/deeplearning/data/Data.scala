package org.mengpan.deeplearning.data

import breeze.linalg.{DenseMatrix, DenseVector}

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
