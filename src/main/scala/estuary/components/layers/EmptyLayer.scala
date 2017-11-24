package estuary.components.layers

import breeze.linalg.DenseMatrix

/**
  * Created by mengpan on 2017/9/7.
  */
object EmptyLayer extends ClassicLayer {

  /** ClassicLayer hyperparameters */
  val numHiddenUnits = 0
  val batchNorm = false

  protected def activate(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] =
    throw new Error("EmptyLayer.activate")

  protected def activateGrad(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] =
    throw new Error("EmptyLayer.activateGrad")

  def copyStructure: ClassicLayer = this

  override def setPreviousHiddenUnits(numHiddenUnits: Int): EmptyLayer.type = EmptyLayer
}
