package org.mengpan.deeplearning.components.optimizer

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{pow, sqrt}
import org.mengpan.deeplearning.components.layers.{DropoutLayer, Layer}
import org.mengpan.deeplearning.utils.{DebugUtils, ResultUtils}

/**
  * Created by mengpan on 2017/9/9.
  */
class SGDOptimizer extends Optimizer{
  override protected var miniBatchSize: Int = _

  def setMiniBatchSize(miniBatchSize: Int): this.type = {
    this.miniBatchSize = miniBatchSize
    this
  }

  def getMiniBatchSize: Int = this.miniBatchSize
}
