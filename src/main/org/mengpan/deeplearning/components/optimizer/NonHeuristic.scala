package org.mengpan.deeplearning.components.optimizer

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.log4j.Logger
import org.mengpan.deeplearning.components.layers.{DropoutLayer, Layer}
import org.mengpan.deeplearning.utils.{DebugUtils, ResultUtils}

/**
  * Created by mengpan on 2017/9/10.
  */
trait NonHeuristic extends Optimizer{
  override val logger = Logger.getLogger(this.getClass)

  def updateParams(paramsList: List[(DenseMatrix[Double], DenseVector[Double])],
                   learningrate: Double,
                   backwardResList: List[ResultUtils.BackwardRes],
                   iterationTime: Int,
                   layers: List[Layer]): NNParams = {

    paramsList
      .zip(backwardResList)
      .zip(layers)
      .map{f =>
        val layer = f._2
        val (w, b) = f._1._1
        val backwardRes = f._1._2

        layer match {
          case _:DropoutLayer => (w, b)
          case _ =>
            val dw = backwardRes.dWCurrent
            val db = backwardRes.dBCurrent

            logger.debug(DebugUtils.matrixShape(w, "w"))
            logger.debug(DebugUtils.matrixShape(dw, "dw"))

            w :-= learningrate * dw
            b :-= learningrate * db

            (w, b)
        }
      }
  }
}
