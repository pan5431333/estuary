package org.mengpan.deeplearning.model

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.abs
import org.apache.log4j.Logger

import scala.collection.mutable

/**
  * Created by mengpan on 2017/8/23.
  */
trait Model {
  val logger = Logger.getLogger("Model")

  var learningRate: Double
  var iterationTime: Int
  val costHistory: mutable.TreeMap[Int, Double] = new mutable.TreeMap[Int, Double]()

  def setLearningRate(learningRate: Double): this.type = {
    this.learningRate = learningRate
    this
  }

  def setIterationTime(iterationTime: Int): this.type = {
    this.iterationTime = iterationTime
    this
  }


  def train(feature: DenseMatrix[Double], label: DenseVector[Double]): this.type

  def predict(feature: DenseMatrix[Double]): DenseVector[Double]

  def accuracy(label: DenseVector[Double], labelPredicted: DenseVector[Double]): Double = {
    val numCorrect = (0 until label.length)
      .map{index =>
        if (abs(label(index) - labelPredicted(index)) < scala.math.pow(10, -6)) 1 else 0
      }
      .count(_ == 1)
    numCorrect.toDouble / label.length.toDouble
  }

  def getCostHistory: mutable.TreeMap[Int, Double] = this.costHistory
  def getLearningRate: Double = this.learningRate
  def getIterationTime: Int = this.iterationTime
}
