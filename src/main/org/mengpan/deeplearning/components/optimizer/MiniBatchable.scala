package org.mengpan.deeplearning.components.optimizer

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.util.Random

/**
  * Created by mengpan on 2017/9/10.
  */
trait MiniBatchable {
  protected var miniBatchSize: Int = _

  def setMiniBatchSize(miniBatchSize: Int): this.type = {
    assert(miniBatchSize > 0, "Minibatch size must be positive. ")

    this.miniBatchSize = miniBatchSize
    this
  }

  def getMiniBatchSize: Int = this.miniBatchSize

  def getMiniBatches(feature: DenseMatrix[Double],
                     label: DenseVector[Double]): List[(DenseMatrix[Double], DenseVector[Double])] = {

    this.miniBatchSize match {
      case a if a > feature.rows => throw new IllegalArgumentException(
        "mini batch size(" + this.miniBatchSize + ")must be less than the number of examples(" + feature.rows + ")!")
      case a if a > 0 => getPositiveNumMiniBatches(feature, label, a)
      case _ => throw new IllegalArgumentException("mini-batch size: " + this.miniBatchSize + " number of exmaples: " + feature.rows)
    }
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
