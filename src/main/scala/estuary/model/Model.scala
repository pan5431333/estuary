package estuary.model

import breeze.linalg.DenseMatrix

trait Model extends ModelLike[Model] {

  def init[Params](rows: Int, cols: Int): Params

  def forwardAndCalCost(feature: DenseMatrix[Double], label: DenseMatrix[Double], params: Seq[DenseMatrix[Double]]): Double

  def backwardWithGivenParams(label: DenseMatrix[Double], params: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]]

  def copyStructure: Model
}
