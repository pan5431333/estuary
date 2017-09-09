package org.mengpan.deeplearning.components.initializer

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Rand
import org.mengpan.deeplearning.components.layers.Layer

/**
  * Created by mengpan on 2017/9/5.
  */
trait WeightsInitializer {
  protected def getWeightsMultipliyer(previousLayerDim: Int, currentLayerDim: Int): Double

  def init(inputDim: Int,
           allLayers: List[Layer]):
  List[(DenseMatrix[Double], DenseVector[Double])] = {
    val layersDims: Vector[Int] = allLayers
      .map(_.numHiddenUnits)
      .::(inputDim)
      .toVector

    val numLayers = layersDims.length

    (1 until numLayers).map{i =>
      val w = DenseMatrix.rand[Double](layersDims(i-1), layersDims(i), rand=Rand.gaussian) * getWeightsMultipliyer(layersDims(i-1), layersDims(i))
      val b = DenseVector.zeros[Double](layersDims(i))
      (w, b)
    }
      .toList
  }
}
