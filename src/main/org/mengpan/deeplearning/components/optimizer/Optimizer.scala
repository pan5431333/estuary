package org.mengpan.deeplearning.components.optimizer

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.log4j.Logger


/**
  * Created by mengpan on 2017/9/9.
  */
trait Optimizer {
  type NNParams = List[(DenseMatrix[Double], DenseVector[Double])]
  val logger = Logger.getLogger(this.getClass)
}
