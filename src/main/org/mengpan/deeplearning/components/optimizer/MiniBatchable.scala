package org.mengpan.deeplearning.components.optimizer

import breeze.linalg.{DenseMatrix, DenseVector, min}

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
                     label: DenseMatrix[Double]): Iterator[(DenseMatrix[Double], DenseMatrix[Double])] = {

    this.miniBatchSize match {
      case a if a > feature.rows => throw new IllegalArgumentException(
        "mini batch size(" + this.miniBatchSize + ")must be less than the number of examples(" + feature.rows + ")!")
      case a if a > 0 => getPositiveNumMiniBatches(feature, label, a)
      case _ => throw new IllegalArgumentException("mini-batch size: " + this.miniBatchSize + " number of exmaples: " + feature.rows)
    }
  }

  private def getPositiveNumMiniBatches(feature: DenseMatrix[Double], label: DenseMatrix[Double], miniBatchSize: Int): Iterator[(DenseMatrix[Double], DenseMatrix[Double])] = {
    val numExamples = feature.rows
    val inputDim = feature.cols
    val outputDim = label.cols
    val shuffledIndex = Random.shuffle[Int, Vector]((0 until numExamples).toVector)
    val numMiniBatchesFloor = numExamples / miniBatchSize
    val isDivided = numExamples % miniBatchSize == 0
    val numMiniBatches = if (isDivided) numMiniBatchesFloor else numMiniBatchesFloor + 1

    (0 until numMiniBatches).toIterator.map{i =>
      val startIndex = i * miniBatchSize
      val endIndex = min((i+1) * miniBatchSize, numExamples)
      val indexes = shuffledIndex.slice(startIndex, endIndex)
      (feature(indexes, ::).toDenseMatrix, label(indexes, ::).toDenseMatrix)
    }
  }
}
