package org.mengpan.deeplearning.utils

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by mengpan on 2017/8/26.
  */
object ResultUtils {
  case class ForwardRes(val yPrevious: DenseMatrix[Double],
                        val zCurrent: DenseMatrix[Double],
                        val yCurrent: DenseMatrix[Double]) {
    override def toString: String = "yPrevious:{" + yPrevious + "}\n" +
    "zCurrent: {" + zCurrent + "}\n" +
    "yCurrent: {" + yCurrent + "}\n"
  }


  case class BackwardRes(val dYPrevious: DenseMatrix[Double],
                         val dWCurrent: DenseMatrix[Double],
                         val dBCurrent: DenseVector[Double]) {
    override def toString: String = "dYPrevious:{" + dYPrevious + "}\n" +
      "dWCurrent: {" + dWCurrent + "}\n" +
      "dBCurrent: {" + dBCurrent + "}\n"
  }
}
