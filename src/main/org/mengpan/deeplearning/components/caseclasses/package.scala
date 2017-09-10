package org.mengpan.deeplearning.components

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by mengpan on 2017/9/10.
  */
package object caseclasses {

  case class AdamOptimizationParams(modelParams: List[(DenseMatrix[Double], DenseVector[Double])],
                                    momentumParams: List[(DenseMatrix[Double], DenseVector[Double])],
                                    adamParams: List[(DenseMatrix[Double], DenseVector[Double])])

}
