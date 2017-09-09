package org.mengpan.deeplearning.components.optimizer

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sqrt
import breeze.numerics.pow
import breeze.stats.distributions.Rand
import org.apache.log4j.Logger
import org.mengpan.deeplearning.components.layers.{DropoutLayer, Layer}
import org.mengpan.deeplearning.components.optimizer.GDOptimizer.{NNParams, logger}
import org.mengpan.deeplearning.utils.{DebugUtils, ResultUtils}

import scala.util.Random

/**
  * Created by mengpan on 2017/9/9.
  */
trait Optimizer {
  type NNParams = List[(DenseMatrix[Double], DenseVector[Double])]
  val logger = Logger.getLogger(this.getClass)

  protected var miniBatchSize: Int

  def updateParams(paramsList: List[(DenseMatrix[Double], DenseVector[Double])],
                    previousMomentum: List[(DenseMatrix[Double], DenseVector[Double])],
                    previousAdam: List[(DenseMatrix[Double], DenseVector[Double])],
                    learningrate: Double,
                    backwardResList: List[ResultUtils.BackwardRes],
                    iterationTime: Int,
                    miniBatchTime: Double,
                    layers: List[Layer]): (NNParams, NNParams, NNParams) = {

    val updatedParams = paramsList
      .zip(backwardResList)
      .zip(layers)
      .zip(previousMomentum)
      .zip(previousAdam)
      .map{f =>
        val layer = f._1._1._2
        val (w, b) = f._1._1._1._1

        val backwardRes = f._1._1._1._2
        val momentum = f._1._2
        val adam = f._2

        layer match {
          case _:DropoutLayer => ((w, b), momentum, adam)
          case _ =>
            val dw = backwardRes.dWCurrent
            val db = backwardRes.dBCurrent

            logger.debug(DebugUtils.matrixShape(w, "w"))
            logger.debug(DebugUtils.matrixShape(dw, "dw"))

            w :-= learningrate * dw
            b :-= learningrate * db

            ((w, b), momentum, adam)
        }
      }

    val (modelParams, momentumAndAdam) = updatedParams
      .unzip(f => (f._1, (f._2, f._3)))

    val (momentum, adam) = momentumAndAdam.unzip(f => (f._1, f._2))

    (modelParams, momentum, adam)
  }

  def getMiniBatches(feature: DenseMatrix[Double],
                     label: DenseVector[Double]): List[(DenseMatrix[Double], DenseVector[Double])] = {

    this.miniBatchSize match {
      case -100 => List((feature, label))
      case a if a > feature.rows => throw new IllegalArgumentException(
        "mini batch size(" + this.miniBatchSize + ")must be less than the number of examples(" + feature.rows + ")!")
      case a if a > 0 => getPositiveNumMiniBatches(feature, label, this.miniBatchSize)
      case _ => throw new IllegalArgumentException("mini-batch size: " + this.miniBatchSize + " number of exmaples: " + feature.rows)
    }
  }


  def init(inputDim: Int, layers: List[Layer]): NNParams = {
    val layersDims: Vector[Int] = layers
      .map(_.numHiddenUnits)
      .::(inputDim)
      .toVector

    val numLayers = layersDims.length

    (1 until numLayers).map{i =>
      val w = DenseMatrix.zeros[Double](layersDims(i-1), layersDims(i))
      val b = DenseVector.zeros[Double](layersDims(i))
      (w, b)
    }
      .toList
  }


  private def getPositiveNumMiniBatches(feature: DenseMatrix[Double], label: DenseVector[Double], miniBatchSize: Int): List[(DenseMatrix[Double], DenseVector[Double])] = {
    val numExamples = feature.rows.toDouble
    val inputDim = feature.cols
    val shuffledIndex = Random.shuffle((0 until numExamples.toInt).toList).toList
    val numMiniBatchesFloor = numExamples / miniBatchSize
    val numMiniBatches = if (numMiniBatchesFloor % 1 < 1E-8) numMiniBatchesFloor.toInt else numMiniBatchesFloor.toInt + 1

    (0 until numMiniBatches)
      .map{List.fill[Int](miniBatchSize)(_)}
      .flatten
      .zip(shuffledIndex)
      .map{f =>
        val (miniBatch, rowIndex) = f
        (miniBatch, feature(rowIndex, ::), label(rowIndex))
      }
      .groupBy(_._1)
      .map{f =>
        val groupData = f._2

        val feature = DenseMatrix.zeros[Double](groupData.length, inputDim)
        val label = DenseVector.zeros[Double](groupData.length)

        (0 until groupData.length).foreach{rowIndex =>
          feature(rowIndex, ::) := groupData(rowIndex)._2
          label(rowIndex) = groupData(rowIndex)._3
        }

        (feature, label)
      }
      .toList
  }

}
