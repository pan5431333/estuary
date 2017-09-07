package org.mengpan.deeplearning.components.regularizer

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{log, pow}
import org.mengpan.deeplearning.components.layers.Layer
import org.mengpan.deeplearning.utils.ResultUtils.{BackwardRes, ForwardRes}

/**
  * Created by mengpan on 2017/9/5.
  */
trait Regularizer {
  def getReguCost(paramsList: List[(DenseMatrix[Double], DenseVector[Double])]): Double
  def getReguCostGrad(w: DenseMatrix[Double], numExamples: Int): DenseMatrix[Double]

  var lambda: Double = 0.7

  def setLambda(lambda: Double): this.type = {
    if (lambda < 0) throw new IllegalArgumentException("Lambda must be nonnegative!")

    this.lambda = lambda
    this
  }
}
