package org.mengpan.deeplearning.components.optimizer

import breeze.linalg.{DenseMatrix, DenseVector}
import org.mengpan.deeplearning.components.layers.Layer

/**
  * Created by mengpan on 2017/9/10.
  */
trait Heuristic extends Optimizer{

  protected var momentumRate: Double
  protected var adamRate: Double

  def setMomentumRate(momentum: Double): this.type = {
    this.momentumRate = momentum
    this
  }

  def setAdamRate(adamRate: Double): this.type = {
    this.adamRate = adamRate
    this
  }
}
