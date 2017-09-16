package org.mengpan.deeplearning.components.optimizer

import breeze.linalg.{DenseMatrix, DenseVector, Matrix}
import org.apache.log4j.Logger
import org.mengpan.deeplearning.components.regularizer.Regularizer

import scala.collection.mutable
import scala.{specialized => spec}


/**
  * Created by mengpan on 2017/9/9.
  */
trait Optimizer {
  val logger = Logger.getLogger(this.getClass)

  var iteration: Int = _
  var learningRate: Double = _
  def setIteration(iteration: Int): this.type = {
    this.iteration = iteration
    this
  }
  def setLearningRate(learningRate: Double): this.type = {
    assert(learningRate > 0, "Nonpositive learning rate: " + learningRate)
    this.learningRate = learningRate
    this
  }

  val costHistory: mutable.MutableList[Double] = new mutable.MutableList[Double]()

  def optimize[T <: Seq[DenseMatrix[Double]]](feature: DenseMatrix[Double], label: DenseMatrix[Double])
                                             (initParams: T)
                                             (forwardFunc: (DenseMatrix[Double], DenseMatrix[Double], T) => Double)
                                             (backwardFunc: (DenseMatrix[Double], T) => T): T
}
