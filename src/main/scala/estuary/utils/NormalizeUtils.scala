package estuary.utils

import breeze.linalg.{DenseMatrix, DenseVector}
import estuary.data.Data
import estuary.helper.DlCollection

/**
  * Created by mengpan on 2017/8/26.
  */
object NormalizeUtils {
  def normalizeBy[E <: Data](data: DlCollection[E])
                            (normalizeFunc: DenseVector[Double]
                              => DenseVector[Double]): DlCollection[E] = {
    val feature = data.getFeatureAsMatrix
    val numCols = feature.cols
    val numRows = feature.rows

    val normalizedFeature = DenseMatrix.zeros[Double](numRows, numCols)

    (0 until numCols).foreach { j =>
      val ithCol = feature(::, j)
      normalizedFeature(::, j) := normalizeFunc(ithCol)
    }

    var i = -1
    val res = data.map[E] { eachData =>
      i += 1
      eachData.updateFeature(normalizedFeature(i, ::).t)
    }

    res
  }

  def normalizeBy(feature: DenseMatrix[Double])
                 (normalizeFunc: DenseVector[Double] => DenseVector[Double]): DenseMatrix[Double] = {
    val numCols = feature.cols
    val numRows = feature.rows
    val normalizedFeature = DenseMatrix.zeros[Double](numRows, numCols)

    (0 until numCols).par.foreach { j =>
      val ithCol = feature(::, j)
      normalizedFeature(::, j) := normalizeFunc(ithCol)
    }

    normalizedFeature
  }
}
