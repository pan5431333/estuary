package estuary.components.optimizer

import breeze.linalg.{DenseMatrix, min}

import scala.util.Random

/**
  * Optimizer using mini-batch optimization algorithm.
  * Its implementations has the ability of "getMiniBatches()"
  */
trait MiniBatchable {
  protected var miniBatchSize: Int = _

  def setMiniBatchSize(miniBatchSize: Int): this.type = {
    assert(miniBatchSize > 0, "Minibatch size must be positive. ")

    this.miniBatchSize = miniBatchSize
    this
  }

  /**
    * Split the whole training set (feature, label) to an iterator of multiple mini-batches.
    *
    * @param feature Feature matrix
    * @param label   Label matrix in one-hot representation
    * @return An iterator of multiple minibatches.
    */
  protected def getMiniBatches(feature: DenseMatrix[Double], label: DenseMatrix[Double]): Iterator[(DenseMatrix[Double], DenseMatrix[Double])] = {
    assert(feature.rows == label.rows, "feature.rows != label.rows")

    this.miniBatchSize match {
      case a if a > feature.rows => throw new IllegalArgumentException(
        "mini batch size(" + this.miniBatchSize + ")must be less than number of examples(" + feature.rows + ")!")
      case a if a == feature.rows => Iterator((feature, label))
      case a if a > 0 => getPositiveNumMiniBatches(feature, label, a)
      case _ => throw new IllegalArgumentException("mini-batch size: " + this.miniBatchSize + " |number of exmaples: " + feature.rows)
    }
  }

  /**
    * Split up a training set into an iterator of multiple mini-batches. Each mini-batch has miniBatchSize# of examples.
    *
    * @param feature       Feature matrix
    * @param label         Label matrix in one-hot representation
    * @param miniBatchSize generally 64, 128, 256, 512, etc.
    * @return An iterator of multiple minibatches.
    */
  private def getPositiveNumMiniBatches(feature: DenseMatrix[Double], label: DenseMatrix[Double], miniBatchSize: Int): Iterator[(DenseMatrix[Double], DenseMatrix[Double])] = {
    val numExamples = feature.rows
    val shuffledIndex = Random.shuffle[Int, Vector]((0 until numExamples).toVector)
    val numMiniBatchesFloor = numExamples / miniBatchSize
    val isDivided = numExamples % miniBatchSize == 0
    val numMiniBatches = if (isDivided) numMiniBatchesFloor else numMiniBatchesFloor + 1

    (0 until numMiniBatches).toIterator.map { i =>
      val startIndex = i * miniBatchSize
      val endIndex = min((i + 1) * miniBatchSize, numExamples)
      val indexes = shuffledIndex.slice(startIndex, endIndex)
      (feature(indexes, ::).toDenseMatrix, label(indexes, ::).toDenseMatrix)
    }
  }
}
