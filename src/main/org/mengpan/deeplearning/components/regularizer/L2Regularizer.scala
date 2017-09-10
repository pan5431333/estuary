package org.mengpan.deeplearning.components.regularizer

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.pow

/**
  * Created by mengpan on 2017/9/5.
  */
class L2Regularizer extends Regularizer{

  override def getReguCost(paramsList: List[(DenseMatrix[Double], DenseVector[Double])]):
  Double = {

    //用foldLeft进行List求和操作，比MapReduce更快，这里是Scala的应用重点
    paramsList
      .foldLeft[Double](0.0){(total, params) =>
      total + sum(pow(params._1, 2)) / 2.0
    }
  }


  override def getReguCostGrad(w: DenseMatrix[Double], numExamples: Int):
  DenseMatrix[Double] = this.lambda * w / numExamples.toDouble
}

object L2Regularizer {
  def apply(lambda: Double): L2Regularizer = {
    new L2Regularizer()
      .setLambda(lambda)
  }
}