package org.mengpan.deeplearning.components

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sqrt

/**
  * Created by mengpan on 2017/9/5.
  */
trait WeightsInitializer {
  protected def getWeightsMultipliyer(previousLayerDim: Int, currentLayerDim: Int): Double

  def init(numExamples: Int, inputDim: Int,
           hiddenLayerStructure: Map[Int, Byte],
           outputLayerStructure: (Int, Byte)):
  List[(DenseMatrix[Double], DenseVector[Double])] = {
    val layersDims: Vector[Int] = Nil
      .::(outputLayerStructure._1)
      .:::(hiddenLayerStructure.map(_._1).toList)
      .::(inputDim)
      .toVector

    val numLayers = layersDims.length

    (1 until numLayers).map{i =>
      val w = DenseMatrix.rand[Double](layersDims(i-1), layersDims(i)) * getWeightsMultipliyer(layersDims(i-1), layersDims(i))
      val b = DenseVector.zeros[Double](layersDims(i))
      (w, b)
    }
      .toList
  }
}
