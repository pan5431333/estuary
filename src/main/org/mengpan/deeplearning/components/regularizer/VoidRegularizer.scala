package org.mengpan.deeplearning.components.regularizer

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by mengpan on 2017/9/5.
  */
object VoidRegularizer extends Regularizer{
  override def getReguCost(paramsList: List[(DenseMatrix[Double], DenseVector[Double])]): Double = 0.0

  override def getReguCostGrad(w: DenseMatrix[Double],
                                         numExamples: Int):
  DenseMatrix[Double] = DenseMatrix.zeros[Double](w.rows, w.cols)
}
