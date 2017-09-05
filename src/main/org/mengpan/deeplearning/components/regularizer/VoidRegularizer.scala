package org.mengpan.deeplearning.components.regularizer

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by mengpan on 2017/9/5.
  */
object VoidRegularizer extends Regularizer{
  override protected def getReguCost(paramsList: List[(DenseMatrix[Double], DenseVector[Double])]): Double = 0.0
}
